suppressMessages({
library("INLA")
library("ngme2")
library("rSPDE")
library("fields")  
})

mesh = inla.mesh.create(
  lattice = inla.mesh.lattice(x = seq(0, 1, length.out = 100), y = seq(0,1,length.out = 100)),
  extend = FALSE, refine = FALSE
)

fem  <- inla.mesh.fem(mesh)

simulate <- function(fem, loc, range, smooth) {
  
  h <- diag(fem$c0)
  
  op <- matern.operators(
    range = range, 
    sigma = 1, 
    nu = smooth, 
    mesh = mesh,
    type = "operator",
    parameterization = "matern"
  )
  
  W = rnig(n = length(h), delta = -10, mu = 10, nu = 100, sigma = 1, h = h)
  Y = as.vector(Pl.solve(op, Pr.mult(op, W)))
  proj = inla.mesh.projector(mesh, loc = loc)
  Z = inla.mesh.project(proj, Y)
  
  return(Z)
}



# Example:
# n   <- 250
# loc <- cbind(x = runif(n), y = runif(n))
# Z   <- simulate(fem, loc, range = 0.2, smooth = 0.7)
# df  <- data.frame(Z = Z, x = loc[, 1], y = loc[, 2])
# ggplot(df) + 
#   geom_point(aes(x = x, y = y, colour = Z)) +
#   scale_color_distiller(palette = "Spectral") +
#   theme_bw() + coord_fixed()


# TODO make a grid of plots that shows realisations from the model (show the 
# process as a completely observed field, not over the observed locations)


