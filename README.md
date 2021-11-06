# Funky attractors

Code for the blog post [Strange attractors and probability theory](https://pierresegonne.github.io/Attractors/)

Are supported the following attractors:

* 2D:
  * [Clifford](https://www.worldscientific.com/worldscibooks/10.1142/2052) (Clifford A. Pickover "The Pattern Book: Fractals, Art, and Nature" World Scientific, ?, 1995.)
  * [De Jong](https://www.worldscientific.com/worldscibooks/10.1142/2052) (Clifford A. Pickover "The Pattern Book: Fractals, Art, and Nature" World Scientific, 197-198, 1995.)
  * [Ikeda](https://www.worldscientific.com/worldscibooks/10.1142/2052) (Clifford A. Pickover, "The Pattern Book: Fractals, Art, and Nature" World Scientific, 64-65, 1995.)
* 3D:
  * [Lorenz](https://mathworld.wolfram.com/LorenzAttractor.html#:~:text=The%20Lorenz%20attractor%20is%20an,thermal%20diffusivity%20%2C%20and%20kinematic%20viscosity%20.) (Lorenz, E. N. "Deterministic Nonperiodic Flow." J. Atmos. Sci. 20, 130-141, 1963.)
  * [Rossler](http://paulbourke.net/fractals/rossler/) (Rössler, O. E. "An Equation for Continuous Chaos" Physics Letters, 57A (5), 397-398, 1976.)

### Examples

<img src="examples/clifford.png" width="200"/>
<img src="examples/de_jong.png" width="200"/>

### Installation

run `pip install -r requirements.txt`

### Running

`python main ${attractor_name} ${...options}` use the `--help` options for more information.