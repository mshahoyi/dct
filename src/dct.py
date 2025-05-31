import functools
from torch import nn, vmap
import torch
from torch.func import vjp, jvp, grad
from torch.nn import functional as F
from tqdm import tqdm
import math
from scipy.optimize import root_scalar


def rgetattr(obj, path):
    return functools.reduce(getattr, path.split("."), obj)
def rhasattr(obj, path):
    try:
        functools.reduce(hasattr, path.split("."), obj)
        return True
    except (AttributeError, TypeError):
        return False


class SlicedModel(nn.Module):
    def __init__(self, model, start_layer, end_layer, layers_name=None):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        if layers_name is None:
            if hasattr(self.model, "layers"):  
                self.layers_name = "model.layers"
            elif hasattr(self.model, "model"): 
                self.layers_name =  "model.model.layers"
            else:
                raise ValueError(f"don't know how to get layer list for {type(model)}")
        else:
            self.layers_name = layers_name
        self.layers = rgetattr(self.model, self.layers_name)
        self.layers_name_split = self.layers_name.split(".")

    def reset(self):
        setattr(self.model.config, "num_hidden_layers",self.depth)
        setattr(rgetattr(self.model, ".".join(self.layers_name_split[:-1])), self.layers_name_split[-1], self.L)
        for i in range(len(rgetattr(self.model, self.layers_name))):
            rgetattr(self.model, self.layers_name)[i].self_attn.layer_idx = i
        pass


    def forward(self, h):
        # mutate model so that forward pass only runs the specified middle layers

        # Original info
        self.L = self.layers
        self.depth = self.model.config.num_hidden_layers

        layers_name_split = self.layers_name_split

        # Sets the layers in the model only to the selected ones
        selected_layers = self.L[self.start_layer:self.end_layer+1]
        setattr(rgetattr(self.model, ".".join(layers_name_split[:-1])), layers_name_split[-1], selected_layers)

        # Sets the number of layers in the model to the number of selected layers
        setattr(self.model.config, "num_hidden_layers",self.end_layer-self.start_layer+1)
        
        # Sets the layer index for each layer
        for i in range(len(rgetattr(self.model, self.layers_name))):
            rgetattr(self.model, self.layers_name)[i].self_attn.layer_idx = i

        # actually run the forward pass
        result = self.model(inputs_embeds=h, output_hidden_states=True).hidden_states[self.end_layer-self.start_layer]

        # reset model to un-mutated state
        self.reset()

        return result

class DeltaActivations(nn.Module):
    def __init__(self, sliced_model, target_position_indices=slice(-3,None)):
        super().__init__()
        self.sliced_model = sliced_model
        self.device = sliced_model.model.device
        self.target_position_indices = target_position_indices
    def forward(self, theta, x, y):
        '''
        computes average delta in target layer activations as a 
        function of bias theta
        '''
        delta = self.sliced_model(x+theta) - y # batch_size x seq_len x d_model
        delta = delta[:, self.target_position_indices, :]
        return delta.mean(dim=1)

class StreamingAverage:
    """
    Maintains a streaming average of tensors.
    Handles variable batch sizes and arbitrary tensor dimensions.
    """
    def __init__(self):
        self.count = 0
        self.mean = None
    
    def update(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Updates the streaming average with a new batch of data.
        
        Args:
            batch: Tensor of shape (batch_size, dim1, ..., dimk)
                  The first dimension is assumed to be the batch dimension
        
        Returns:
            Current mean after incorporating the new batch
        """
        batch_size = batch.size(0)
        
        if self.mean is None:
            # First batch - initialize mean with the correct shape
            self.mean = batch.mean(dim=0)
            self.count = batch_size
            return self.mean
        
        # Update count
        new_count = self.count + batch_size
        
        # Compute batch mean
        batch_mean = batch.mean(dim=0)
        
        # Update mean using formula:
        # new_mean = old_mean + (batch_mean - old_mean) * (batch_size / new_count)
        self.mean = self.mean + (batch_mean - self.mean) * (batch_size / new_count)
        self.count = new_count
        
        return self.mean
    
    def get_mean(self) -> torch.Tensor:
        """Returns the current mean."""
        if self.mean is None:
            raise ValueError("No data has been processed yet")
        return self.mean
    
    def reset(self):
        """Resets the streaming average."""
        self.count = 0
        self.mean = None

class SteeringCalibrator():
    def __init__(self, target_ratio=.5):
        self.target_ratio=target_ratio
    def calibrate(self, delta_acts_single, X, Y, batch_size=1,
                  calibration_sample_size=30, factor_batch_size=16):
        delta_acts = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2,
                  chunk_size=factor_batch_size)
        d_model = X.shape[2]
        V_cal = F.normalize(torch.randn(d_model, calibration_sample_size), dim=0)
        def jvp_single(v,X,Y):
            v0 = torch.zeros(v.shape)
            _, jvp_out = jvp(lambda _v: delta_acts_single(_v,X,Y), (v0,), (v,))
            return jvp_out
        jvp_batch = vmap(lambda v, X, Y: jvp_single(v,X,Y), in_dims=(1,None,None), out_dims=(2), chunk_size=factor_batch_size)

        U_cal_avg = StreamingAverage()
        with torch.no_grad():
            for b in range(0, X.shape[0], batch_size):
                x = X[b:b+batch_size,:,:].to(delta_acts_single.device)
                y = Y[b:b+batch_size,:,:].to(delta_acts_single.device)
                U_cal_batch = jvp_batch(V_cal, x, y)
                U_cal_avg.update(U_cal_batch)
        U_cal = U_cal_avg.get_mean()
        U_cal_norms = U_cal.norm(dim=0)

        def jacobian_ratio(r):
            denom = (r*U_cal_norms).pow(2)
            delta_acts_avg = StreamingAverage()
            with torch.no_grad():
                for b in range(0,X.shape[0],batch_size):
                    x = X[b:b+batch_size,:,:].to(delta_acts_single.device)
                    y = Y[b:b+batch_size,:,:].to(delta_acts_single.device)
                    delta_acts_batch = delta_acts(r*V_cal, x, y)
                    delta_acts_avg.update(delta_acts_batch)
            num = (delta_acts_avg.get_mean()-r*U_cal).pow(2).sum(dim=0)
            return math.sqrt((num / denom).mean())

        # solve for jacobian_ratio = target_ratio
        soln = root_scalar(lambda r: jacobian_ratio(r)-self.target_ratio, bracket=[.001, 100.0])
        self.R = soln.root
        return self.R
class LinearDCT():
    def __init__(self, num_factors=512):
        self.num_factors=num_factors
        pass
        
    def fit(self, delta_acts_single, X, Y, method="projected", batch_size=1, dim_output_projection=32, factor_batch_size=16, input_scale=1.0):
        assert method in ["full","projected"]
        delta_acts = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2,
                  chunk_size=factor_batch_size)
        d_model = X.shape[2]

        if method == "projected":
            # vector-jacobian product (backwards AD) helper functions
            def vjp_single(u,v,X,Y):
                output, vjp_fn = vjp(lambda _v: delta_acts_single(_v,X,Y), v)
                with torch.no_grad():
                    udots = output @ u
                return udots, output.detach(), vjp_fn(u.expand(X.shape[0], -1))[0].detach()
            vjp_batch = vmap(lambda u,v, X, Y: vjp_single(u,v,X,Y), in_dims=(1,1,None,None),
                             out_dims=(1,2,1), chunk_size=factor_batch_size)
        elif method == "full":
            v0 = torch.zeros(d_model)
            def jvp_single(v,X,Y):
                with torch.no_grad():
                    output, jvp_out = jvp(lambda _v: delta_acts_single(_v, X, Y), (v0,),(v,))
                return jvp_out # d_model
            jvp_batch = vmap(lambda v, X, Y: jvp_single(v,X,Y), in_dims=(1,None,None),
                             out_dims=2,chunk_size=factor_batch_size)
            
        if method=="projected":
            # if projected we will calculate VJPs at random output directions
            U_rand = F.normalize(torch.randn(d_model, dim_output_projection),dim=0)
        else:
            # otherwise use all output directions in standard basis
            dim_output_projection = d_model
            V_in = torch.eye(d_model)

        # will calculate jacobian at zero
        V0 = torch.zeros(d_model, dim_output_projection)

        # loop over data
        print("computing jacobian...")
        J_avg = StreamingAverage()
        with torch.no_grad():
            for t in tqdm(range(0, X.shape[0], batch_size)):
                x = X[t:t+batch_size,:,:].to(delta_acts_single.device)
                y = Y[t:t+batch_size,:,:].to(delta_acts_single.device)
                if method == "projected":
                    _, _, J_batch = vjp_batch(U_rand, V0, x, y)
                    J_batch = J_batch.t()
                    J_avg.update(J_batch.unsqueeze(0))
                elif method == "full":
                    J_batch = jvp_batch(V_in, x, y)
                    J_avg.update(J_batch)
        J = J_avg.get_mean()
        self.J = J

        # compute SVD to get factorization of Jacobian
        print("computing SVD of jacobian...")
        self.U, _, self.V = torch.linalg.svd(J)
        self.V = self.V[:,:self.num_factors]

        # if method=="projected" then we need an extra forward pass to get output directions in full space
        if method=="projected":
            U_avg = StreamingAverage()
            print("computing output directions...")
            with torch.no_grad():
                for b in tqdm(range(0, X.shape[0], batch_size)):
                    x = X[b:b+batch_size,:,:].to(delta_acts_single.device)
                    y = Y[b:b+batch_size,:,:].to(delta_acts_single.device)
                    U_batch = delta_acts(input_scale*self.V, x, y)
                    U_avg.update(U_batch)
            self.U = U_avg.get_mean()
            self.U = F.normalize(self.U, dim=0)
        return self.U, self.V

class QuadraticDCT():
    def __init__(self, num_factors=512):
        self.num_factors=num_factors
        pass

    def _init_rand(self, delta_acts, X, Y):
        print("initializing V,U...")
        # initialize V randomly
        self.V = F.normalize(torch.randn(self.d_source, self.num_factors, device=self.device), dim=0)

        # initialize U as average of delta_acts
        U_avg = StreamingAverage()
        print("computing output directions...")
        with torch.no_grad():
            for b in tqdm(range(0, X.shape[0], batch_size)):
                x = X[b:b+batch_size,:,:].to(self.device)
                y = Y[b:b+batch_size,:,:].to(self.device)
                U_batch = delta_acts(self.V, x, y)
                U_avg.update(U_batch)
        self.U = U_avg.get_mean()
        self.U = F.normalize(self.U, dim=0)
        pass
    def _init_jacobian(self, delta_acts_single, X, Y):
        self.linear_dct = LinearDCT(num_factors = self.num_factors)
        self.U, self.V = self.linear_dct.fit(delta_acts_single, X, Y,
                                             method="projected",dim_output_projection=self.d_proj, 
                                             batch_size=self.batch_size,factor_batch_size=self.factor_batch_size)
        pass
        
        
    def fit(self, delta_acts_single, X, Y, batch_size=1, factor_batch_size=16, init="jacobian", d_proj=32,
            max_iters=20, compute_intermediate_objective=False):
        assert(init in ["random","jacobian"])
        self.num_samples, self.seq_len, self.d_source = X.shape
        self.batch_size = batch_size
        self.factor_batch_size = factor_batch_size
        self.device = delta_acts_single.device
        self.d_proj = d_proj
        self.max_iters = max_iters
        delta_acts = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2,
                  chunk_size=factor_batch_size)
        # init
        if init == "random":
            self._init_rand(delta_acts,X,Y)
        elif init == "jacobian":
            self._init_jacobian(delta_acts_single,X,Y)

        # define autograd functions
        # u'Jv
        def ujv_fn(u,v,X,Y):
            v0 = torch.zeros(self.d_source)
            _, jvp_out = jvp(lambda _v: delta_acts_single(_v, X, Y), (v0,), (v,))
            return (jvp_out @ u).mean()
        # H(u,v,:)
        def huv_single(u,v,X,Y):
            return grad(lambda v: ujv_fn(u,v,X,Y))(v).detach()
        def huv_batch(U,V,X,Y):
            hfunc = vmap(lambda u,v: huv_single(u,v,X,Y), chunk_size=self.factor_batch_size,
                     in_dims=(1,1), out_dims=1)
            return hfunc(U,V)
        # Jv
        def jv1(v, X, Y):
            v0 = torch.zeros(self.d_source)
            return jvp(lambda v_: delta_acts_single(v_,X,Y), (v0,),(v,))[1]
        # H(:,v,v)
        def hvv(v1,v2,X,Y):
            return jvp(lambda _v: jv1(_v,X,Y),(v1,),(v2,))[1].detach().mean(0)
        def hvv_batch(V1,V2,X,Y):
            hfunc = vmap(lambda _v1,_v2: hvv(_v1,_v2,X,Y), chunk_size=self.factor_batch_size,
                         in_dims=(1,1), out_dims=1)
            return hfunc(V1,V2)

        # main training loop
        self.U = nn.Parameter(self.U)
        self.V = nn.Parameter(self.V)
        fdots = []
        objective_values = []
        for i in tqdm(range(max_iters)):
            # orthogonalize
            with torch.no_grad():
                self.V.data, _ = torch.linalg.qr(self.V)

            # loop over data to compute updates
            fdot_avg = StreamingAverage()
            G_U_avg = StreamingAverage()
            G_V_avg = StreamingAverage()
            for b in tqdm(range(0, self.num_samples, self.batch_size)):
                x = X[b:b+self.batch_size,:,:].to(self.device)
                y = Y[b:b+self.batch_size,:,:].to(self.device)
                nb = x.shape[0]
                gvb = huv_batch(self.U,self.V,x,y)
                gub = hvv_batch(self.U,self.V,x,y)
                with torch.no_grad():
                    fb = torch.einsum("if,if->", gub, self.U)
                    G_U_avg.update(gub.unsqueeze(0).expand(nb,-1,-1))
                    G_V_avg.update(gvb.unsqueeze(0).expand(nb,-1,-1))
                    fdot_avg.update(fb.unsqueeze(0).unsqueeze(0).expand(nb,-1))

            # update
            with torch.no_grad():
                G_U = G_U_avg.get_mean()
                G_V = G_V_avg.get_mean()
                self.U.data = F.normalize(G_U, dim=0)
                self.V.data = F.normalize(G_V, dim=0)
                fdots.append(fdot_avg.get_mean()[0].item())
        self.objective_values = fdots
        return self.U, self.V

class ExponentialDCT():
    def __init__(self, num_factors=512):
        self.num_factors = num_factors
    def _init_rand(self, delta_acts, X, Y):
        print("initializing V,U...")
        # initialize V randomly
        self.V = F.normalize(torch.randn(self.d_source, self.num_factors, device=self.device), dim=0)
        self.U = F.normalize(torch.randn(self.d_target, self.num_factors, device=self.device), dim=0)
        pass

    def _init_jacobian(self, delta_acts_single, X, Y):
        self.linear_dct = LinearDCT(num_factors = self.num_factors)
        self.U, self.V = self.linear_dct.fit(delta_acts_single, X, Y, method="projected", dim_output_projection=self.d_proj, 
                                             batch_size=self.batch_size,factor_batch_size=self.factor_batch_size, 
                                             input_scale=self.input_scale)
        pass                

    def rank(self, delta_acts_single, X, Y, target_vec=None, batch_size=1, factor_batch_size=16):
        delta_acts = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2,
                  chunk_size=factor_batch_size)
        num_samples = X.shape[0]
        Delta_avg = StreamingAverage()
        with torch.no_grad():
            for b in tqdm(range(0, num_samples, batch_size)):
                x = X[b:b+self.batch_size,:,:].to(self.device)
                y = Y[b:b+self.batch_size,:,:].to(self.device)
                Delta_batch = delta_acts(self.input_scale * self.V, x, y)              
                Delta_avg.update(Delta_batch)
            if target_vec is None:
                self.alphas = (Delta_avg.get_mean() * self.U).sum(dim=0)
                K = (self.U.t() @ self.U) * torch.expm1(self.V.t() @ self.V)
                self.alphas = torch.linalg.solve(K, self.alphas)
                self.scores = self.alphas.pow(2)
                self.scores, self.indices = torch.sort(self.scores, descending=True)
            else:
                self.scores = Delta_avg.get_mean().t() @ target_vec.to(self.device)
                self.scores, self.indices = torch.sort(self.scores, descending=True)                
        return self.scores, self.indices
        
    def fit(self, delta_acts_single, X, Y, batch_size=1, factor_batch_size=16, init="random", d_proj=32,
            input_scale=1.0, max_iters=10, beta=1.0):
        '''Fit DCT

        Parameters
        ----------
        delta_acts_single : function computing change in target activations as a function of source-layer bias
        (theta, x, y)

        X (tensor) : tensor of source activations (n_samples, d_source)
        
        Y (tensor) : tensor of target activations (n_samples, d_target)
        
        batch_size (int) : batch size over samples
        
        factor_batch_size (int) : batch size over factors

        init (string): initialization strategy {"random", "jacobian"}

        d_proj (int) : dimensionality of jacobian projection (if used for init)

        input_scale (float) : norm of steering vector in source layer

        max_iters (int) : max iters

        beta (float) : default = 1.0, set smaller for more stable training

        '''
        assert(init in ["random","jacobian"])
        self.num_samples, self.seq_len, self.d_source = X.shape
        _, _, self.d_target = Y.shape
        self.batch_size = batch_size
        self.factor_batch_size = factor_batch_size
        self.device = delta_acts_single.device
        self.input_scale = input_scale
        self.d_proj = d_proj
        self.max_iters = max_iters
        self.beta = beta
        delta_acts = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2,
                  chunk_size=factor_batch_size)


        # init
        if init == "random":
            self._init_rand(delta_acts,X,Y)
        elif init == "jacobian":
            self._init_jacobian(delta_acts_single,X,Y)

        # vjp helper functions
        def vjp_single(u,v,X,Y):
            output, vjp_fn = vjp(lambda _v: delta_acts_single(_v,X,Y), v)
            with torch.no_grad():
                udots = output @ u
            return udots, output.detach(), vjp_fn(u.expand(X.shape[0], -1))[0].detach()
        vjp_batch = vmap(lambda u,v, X, Y: vjp_single(u,v,X,Y), in_dims=(1,1,None,None),
                         out_dims=(1,2,1), chunk_size=self.factor_batch_size)

        # main training loop
        self.U = nn.Parameter(self.U)
        self.V = nn.Parameter(self.V)
        fdots = []
        penalties = []
        objective_values = []
        print("training...")
        for i in tqdm(range(self.max_iters)):
            # orthogonalize
            with torch.no_grad():
                self.V.data, _ = torch.linalg.qr(self.V)

            # loop over data to compute updates
            fdot_avg = StreamingAverage()
            G_U_avg = StreamingAverage()
            G_V_avg = StreamingAverage()
            for b in tqdm(range(0, self.num_samples, self.batch_size)):
                x = X[b:b+self.batch_size,:,:].to(self.device)
                y = Y[b:b+self.batch_size,:,:].to(self.device)
                with torch.no_grad():
                    fb, gub, gvb = vjp_batch(self.U, self.input_scale*self.V, x, y)
                    gvb *= self.input_scale
                    G_U_avg.update(gub)
                    G_V_avg.update(gvb.unsqueeze(0))
                    fdot_avg.update(fb.mean(1).unsqueeze(1))
            fdot_all = fdot_avg.get_mean()[0].item()
            fdots.append(fdot_all)
            G_U = G_U_avg.get_mean()
            G_V = G_V_avg.get_mean()
        
            # update
            with torch.no_grad():
                self.U.data = F.normalize(self.beta*G_U+(1-self.beta)*self.U.data, dim=0)
                self.V.data = F.normalize(self.beta*G_V+(1-self.beta)*self.V.data, dim=0)
                objective_values.append(fdot_all)

        self.objective_values = objective_values    
        return self.U, self.V

class ModelEditor():
    def __init__(self, model, mlp_out_name=None, attn_out_name=None, layers_name=None):
        '''
        Note: this will mutate `model`. To reset to original state call restore()
        '''
        self.model = model
        if layers_name is None:
            if hasattr(self.model, "layers"):  
                self.layers_name = "model.layers"
            elif hasattr(self.model, "model"):  # mistral-like
                self.layers_name =  "model.model.layers"
            else:
                raise ValueError(f"don't know how to get layer list for {type(model)}")
        else:
            self.layers_name = layers_name
        self.layers = rgetattr(self.model, self.layers_name)
        pass
        if mlp_out_name is None:
            if rhasattr(self.layers[0], "mlp.down_proj"):
                self.mlp_out_name = "mlp.down_proj"
            else:
                raise ValueError(f"don't know how to get mlp out")
        else:
            self.mlp_out_name = mlp_out_name
        if attn_out_name is None:
            if rhasattr(self.layers[0], "self_attn.o_proj"):
                self.attn_out_name = "self_attn.o_proj"
            else:
                raise ValueError(f"don't know how to get attn out")
        else:
            self.attn_out_name = attn_out_name

        self.module_names = {"mlp.out":self.mlp_out_name, "attn.out":self.attn_out_name}

        self.steered_layers = {}
        self.ablated_modules = {}
        pass

    def steer(self, vec, layer_idx, module="mlp.out"):
        module_name = self.module_names[module]

        # store existing bias if it exists
        if (layer_idx, module_name) not in self.steered_layers:
            bias = rgetattr(self.layers[layer_idx], module_name).bias
            if bias is not None:
                bias = bias.clone()
            self.steered_layers[(layer_idx, module_name)] = bias

        # set bias to vec
        module_obj = rgetattr(self.layers[layer_idx], module_name)
        module_obj.bias = nn.Parameter(vec.to(module_obj.weight.device))
        pass

    def ablate(self, vec, layer_idxs=None, modules=["mlp.out","attn.out"]):
        vec = F.normalize(vec, dim=0)
        if layer_idxs is None:
            layer_idxs = range(self.model.config.num_hidden_layers)
        for i in layer_idxs:
            for module in modules:
                if (i, module) in self.ablated_modules:
                    raise Error("multiple ablations not yet supported")
                module_obj = rgetattr(self.layers[i], self.module_names[module])
                with torch.no_grad():
                    # ablate weight matrix
                    vec = vec.to(module_obj.weight.device)
                    left_mult = vec.t() @ module_obj.weight.data
                    module_obj.weight.data -= torch.einsum("i,j->ij", vec, left_mult)
                    # store vec, left_mult to allow restoring the weight matrix
                    self.ablated_modules[(i,module)] = (vec, left_mult)
        pass
    def restore(self):
        for (layer_idx, module_name), bias in list(self.steered_layers.items()):
            module_obj = rgetattr(self.layers[layer_idx], module_name)
            module_obj.bias = bias
            del self.steered_layers[(layer_idx, module_name)]
        for (layer_idx, module), (vec, left_mult) in list(self.ablated_modules.items()):
            module_obj = rgetattr(self.layers[layer_idx], self.module_names[module])
            with torch.no_grad():
                module_obj.weight.data += torch.einsum("i,j->ij", vec, left_mult)
            del self.ablated_modules[(layer_idx, module)]
        pass
        
