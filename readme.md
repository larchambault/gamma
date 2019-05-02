# Gamma calculation

An implementation of the local $\gamma$ calculation algorithm described by Chen et al:

> Chen, M., Lu, W., Chen, Q., Ruchala, K., & Olivera, G. (2009).
> Efficient gamma index calculation using fast Euclidean distance transform. Physics in Medicine and Biology, 54(7), 2037–2047.
> https://doi.org/10.1088/0031-9155/54/7/012

That calculation uses the euclidean distance transform (EDT) on a $k+1$ space composed of $k$ spatial coordinates (currently $k=3$)
and one dose coordinate. By scaling the the spatial and dose axis with their respective criterion, the gamma value of a point
$(\mathbf x,d)$ on a testing dose distribution is simply the value at the same location on the EDT of the reference dose distribution.

## Dependencies

- `numpy`
- `scipy`
- `scikit-fmm`
   - Installed through anaconda or pip
   - [github repository](https://github.com/scikit-fmm/scikit-fmm)

## Example of use

Assuming two 3D dose maps: `ref` and `test`

```python
from gamma3D import gamma

Delta = 3 # [% of max]
delta = 3 # [mm]

g = gamma(Delta,delta)
g.set_reference(ref)
gamma_map = g(test)
```

## To do:
- [ ] Accelerate with masked arrays and `skfmm.distance`
- [ ] Better management of units
- [ ] allow for sub-sampling or oversampling dose grid within the gamma class
- [ ] dose threshold to ignore low-dose voxels (might require another librairy such as `skfmm`)
- [ ] Allow 2D gamma
- [ ] option for local gamma
