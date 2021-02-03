# 2D-topology-finder
Determines the topology of a 2D surface embedded in d-dimensional
Euclidean space from a dense collection of points on the surface.

## Background
The topology of compact 2D surfaces is completely classified by their
orientability, genus, and number of holes.

Numerically, these three can be determined from a triangulation of the surface.

from [Herbert Edelsbrunner, CPS296.1: COMPUTATIONAL TOPOLOGY](https://www2.cs.duke.edu/courses/fall06/cps296.1/),
> ... suggests an easy algorithm to recognize a compact
> 2-manifold given by its triangulation. First search all triangles
> and orient them consistently as you go until you either succeed,
> establishing orientability, or you encounter a contradiction,
> establishing non-orientability. Thereafter count the vertices,
> edges, and triangles, and the alternating sum uniquely identifies
> the 2-manifold if there is no boundary. Else count the holes, this
> time by searching the edges that belong to only one triangle each.
> For each additional hole the Euler characteristic chi decreases by one,
> giving chi = 2 - 2 g - h in the orientable case and chi = 2 - g - h
> in the non-orientable case. The genus, g, and the number of holes, h,
> identify a unique 2-manifold with boundary within the orientable
> and the non-orientable classes. \
> https://www2.cs.duke.edu/courses/fall06/cps296.1/Lectures/sec-II-1.pdf

## Triangulation algorithm

The triangulation of a collection of points on the surface
is found by the following simple algorithm:

1. We start our iteration from a random point `i` and find the closest ~32 points.
2. Then we find the best fitting plane of these closest points through a
singular-value decomposition, and project them on to this best fitting plane.
3. From these projected points we construct a Delaunay triangulation. Only
the edges and triangles adjacent to `i` we add to the manifold triangulation.
4. Those points adjacent to `i` are added to the queue, and the above three
steps are repeated for all points in the queue. The only difference is that now
we have to carry out a constrained Delaunay triangulation to ensure that we
respect the edges of the previous local-plane triangulations.

Said more simply, our algorithm simply patches up the various Delaunay triangulations
of the local planes to one global triangulation of the manifold.

## Description of files

In the script `triangulation_finder.py`, we implement the function
`triangulation_finder` that from a dense collection of points first
finds a triangulation of the given 2D surface (embedded in d-dimensional
Euclidean space), and then prints out its orientability, genus,
and number of holes.

The conventions used in specifying the mesh (vertices, edges, and
triangles) are given in the string `mesh_convention`.

The utility functions `refine_mesh` and `relax_mesh` may also
be of use when refining and relaxing meshes of 2D manifolds.

The function `create_sphere` is included for a demonstration
of the utility functions. See the jupyter notebook
`triangulation_demonstration.ipynb` for examples
of applications of `triangulation_finder`.
