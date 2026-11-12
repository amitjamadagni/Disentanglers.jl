# [Utilities](@id Utilities)
Present the different utility functions used across the package. We split them into different sections based on how they might be used.

## Optimal isometry

To apply an optimal isometry from the optimization routines, we can invoke the [`apply_optimized_isometry`](@ref apply_optimized_isometry) function.

```@docs
Disentanglers.apply_optimized_isometry
```

```@raw html
<br style="line-height: 150%;"/>
```

## Objective functions related

Riemannian optimizers employ objective functions in order to obtain an optimal isometry. We categorize these objective functions into entropy based and overlap based functions.

### Entropy
```@docs
Disentanglers.alpha_renyi_entropy
Disentanglers.second_renyi_entropy
```

### Entropy gradients
```@docs
Disentanglers.grad_mat_are
Disentanglers.grad_mat_sre
```

### Overlap/Distance measures
```@docs
Disentanglers.unitary_overlap
```

### Overlap/Distance gradients
```@docs
Disentanglers.grad_mat_uo
```


### State overlap
```@docs
Disentanglers.permute_state_and_return_raw_data
Disentanglers.inner_custom
```

### Operator overlap/distance
```@docs
Disentanglers.permute_op_and_return_raw_data
Disentanglers.normalized_Frobenius_norm_opt
```

```@raw html
<br style="line-height: 150%;"/>
```

## Other functions

### Entropy functions
```@docs
Disentanglers.alpha_renyi_cutoff
```
