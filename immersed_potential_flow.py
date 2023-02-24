#! /usr/bin/env python3
#
# Authors: Clemens Verhoosel, Sai Divi
#
# This example demonstrates the immersed (isogeometric) finite element method
# for the simulation of a two-dimensional non-lifting potential flow over a
# cylinder. A rectangular ambient domain of size L×L is considered, from which
# a circle of radius R is trimmed, resulting in the physical domain Ω. Neumann
# conditions are imposed on all boundaries in accordance with the exact
# solution. A zero-average potential constraint is added using a Lagrange
# multiplier to yield the (well-posed) boundary value problem:
#
#   Δφ   = 0           in Ω
#   ∇φ⋅n = ∇φexact⋅n   on ∂Ω
#   ∫φ   = ∫φexact

from nutils import mesh, cli, function, solver, topology, elementseq, transformseq, export, testing
import treelog, numpy, numpy.linalg
from matplotlib import collections, colors

def main(uinf:float, L:float, R:float, nelems:int, degree:int, maxrefine:int):

    '''
    Immersed analysis of a non-lifting potential flow over a cylinder.

    .. arguments::

        uinf [2.0]
            Free stream velocity.
        L [2.0]
            Domain size.
        R [0.5]
            Cylinder radius.
        nelems [7]
            Number of elements.
        degree [2]
            B-spline degree.
        maxrefine [3]
            Trimming bisectioning depth.
    '''

    # Construct the regular ambient mesh
    ambient_domain, geom = mesh.rectilinear([numpy.linspace(-L/2, L/2, nelems+1)]*2)

    # Trim out the circle
    domain = ambient_domain.trim(function.norm2(geom)-R, maxrefine=maxrefine)

    # Initialize the namespace
    ns = function.Namespace()
    ns.R    = R
    ns.x    = geom
    ns.uinf = uinf

    # Construct function space for the potential field
    ns.basis = domain.basis('spline', degree=degree)

    ns.φ   = 'basis_n ?lhs_n'
    ns.u_i = 'φ_,i'

    # Set the exact solution (with zero average)
    ns.φexact = 'uinf x_0 ( 1 + R^2 / (x_0^2 + x_1^2) )'
    ns.φerror = 'φ - φexact'

    # Construct the residual vector
    res_φ  = domain.integral('basis_n,i φ_,i d:x' @ ns, degree=degree*2)
    res_φ -= domain.boundary.integral('basis_n φexact_,i n_i d:x' @ ns, degree=degree*2)

    # Lagrange-multiplier constraint (∫φ=∫φexact)
    res_φ += domain.integral('?λ basis_n d:x' @ ns, degree=degree*2)
    res_λ  = domain.integral('(φ - φexact) d:x' @ ns, degree=degree*2)

    # Solve the constrained system
    sol = solver.solve_linear(['lhs','λ'], [res_φ, res_λ])
    ns  = ns(**sol)

    # Evaluate the error
    L2err, H1err, E, Eexact = numpy.sqrt(domain.integrate(['φerror φerror d:x',
                                                           '(φerror φerror + φerror_,i φerror_,i) d:x',
                                                           '0.5 φ_,i φ_,i d:x',
                                                           '0.5 φexact_,i φexact_,i d:x']@ns, degree=degree*4))

    Eerr = abs(E-Eexact)
    treelog.user(f'errors: L2={L2err:.2e}, H1={H1err:.2e}, energy={Eerr:.2e}')

    # Post-processing
    makeplots(domain, maxrefine, ns, nelems)

    return sol, H1err

# Post-processing function
def makeplots(domain, maxrefine, ns, nelems):

    # Extract the sub-cell topology (implemented below)
    subcell_topology = get_subcell_topo(domain)

    # Velocity field evaluation
    bezier = subcell_topology.sample('bezier', 3)
    points, φvals, uvals = bezier.eval(['x_i', 'φ', 'u_i / uinf']@ns)

    center = subcell_topology.sample('uniform', 1)
    cpoints, cvals = center.eval(['x_i', 'u_i / uinf']@ns)

    # Defining Paul Tol's `rainbow_PuBr` color map [https://personal.sron.nl/~pault/]
    clrs = ['#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
            '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
            '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
            '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
            '#DF4828', '#DA2222', '#B8221E', '#95211B', '#721E17',
            '#521A13']
    cmap = colors.LinearSegmentedColormap.from_list('rainbow_PuBr', clrs)
    cmap.set_bad('#FFFFFF')

    with export.mplfigure('potential.png') as fig:
        ax = fig.add_subplot(111, aspect='equal', title='potential')
        ax.autoscale(enable=True, axis='both', tight=True)
        im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, φvals, shading='gouraud', cmap=cmap)
        ax.add_collection(collections.LineCollection(points[bezier.hull], colors='k', linewidth=0.5, alpha=0.5))
        fig.colorbar(im, label='φ')

    with export.mplfigure('velocity.png') as fig:
        ax = fig.add_subplot(111, aspect='equal', title='velocity')
        ax.autoscale(enable=True, axis='both', tight=True)
        im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, numpy.linalg.norm(uvals, axis=1), shading='gouraud', cmap=cmap)
        ax.quiver(cpoints[:,0], cpoints[:,1], cvals[:,0], cvals[:,1], scale=10*nelems/numpy.linalg.norm(cvals, axis=1), scale_units='width')
        ax.add_collection(collections.LineCollection(points[bezier.hull], colors='k', linewidth=0.5, alpha=0.5))
        fig.colorbar(im, label='V/Vinf')

# Get integration sub-cell topology
def get_subcell_topo(domain):
  references = []
  transforms = []
  opposites  = []
  for eref, etr, eopp in zip(domain.references, domain.transforms, domain.opposites):
    for tr, ref in eref.simplices:
      references.append(ref)
      transforms.append(etr+(tr,))
      opposites.append(eopp+(tr,))
  references = elementseq.References.from_iter(references, domain.ndims)
  opposites  = transformseq.PlainTransforms(opposites, todims=domain.ndims, fromdims=domain.ndims)
  transforms = transformseq.PlainTransforms(transforms, todims=domain.ndims, fromdims=domain.ndims)
  return topology.TransformChainsTopology('X', references, transforms, opposites)

if __name__=='__main__':
    cli.run(main)

class test(testing.TestCase):

    def test_p1(self) :
        args, H1err = main(uinf=2.0, L=2.0, R=0.5, nelems=4, maxrefine=3, degree=1)
        with self.subTest('lambda'):
            self.assertAlmostEqual(args['λ'], 0)
        with self.subTest('lhs'):
            self.assertAlmostEqual64(args['lhs'], '''
                eNozPC54nA+IDY93nxA6UX1c6ET3CQY4KDV/Z95q8c681Py8xXuLT0B83gIA9FUT9w==''')
        with self.subTest('err'):
            self.assertAlmostEqual(H1err, 0.9546445145978546, places=6)

    def test_p2(self):
        args, H1err = main(uinf=2.0, L=2.0, R=0.5, nelems=4, maxrefine=3, degree=2)
        with self.subTest('lambda'):
            self.assertAlmostEqual(args['λ'], 0)
        with self.subTest('lhs'):
            self.assertAlmostEqual64(args['lhs'],'''
                eNqTPc53/O2xt8f4jssen3287rgbENYBWR0nj5/cdHzT8eMnO05WmFma+Vn4WViaVZilWjRZ7ALCJotU
                i8cWnyyELYUtPwFZAKb/I+8=''')
        with self.subTest('H1err'):
            self.assertAlmostEqual(H1err, 0.42218496186521309, places=6)
