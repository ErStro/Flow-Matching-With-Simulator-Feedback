import jax
import jax.numpy as jnp
from jax import grad

class LotkaVolterraLoss:
    def __init__(self, x_obs: jnp.ndarray):
        """
        Initialisiert die Klasse mit Beobachtungsdaten.
        
        Args:
            x_obs: jnp.ndarray mit 20 Werten: zuerst 10 Prey, dann 10 Predator
        """
        assert x_obs.shape[0] == 20, "x_obs must have 20 values (10 prey + 10 predator)"
        self.x_obs = x_obs
        self.prey_obs = x_obs[:10]
        self.predator_obs = x_obs[10:]
        self.t = jnp.linspace(0, 500, 20)
        self.y0 = jnp.array([5.0, 30.0])

    def simulate(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Führt die Lotka-Volterra-Simulation durch.
        
        Args:
            theta: jnp.ndarray der Form [α, β, δ, γ]
        
        Returns:
            jnp.ndarray der Form (len(t), 2) mit Spalten [prey, predator]
        """
        total_time = self.t[-1]
        steps = 20000
        dt = total_time / steps
        timesteps = jnp.arange(steps + 1) * dt

        alpha, beta, delta, gamma = theta

        def rhs(state, _):
            x, y = state
            dx = alpha * x - beta * x * y
            dy = -gamma * y + delta * x * y
            x_new = x + dt * dx
            y_new = y + dt * dy
            return (x_new, y_new), (x_new, y_new)

        init_state = (self.y0[0], self.y0[1])
        _, states = jax.lax.scan(rhs, init_state, None, steps)

        prey_full = jnp.concatenate([jnp.array([self.y0[0]]), states[0]])
        predator_full = jnp.concatenate([jnp.array([self.y0[1]]), states[1]])

        idxs = jnp.searchsorted(timesteps, self.t)
        prey_out = prey_full[idxs]
        predator_out = predator_full[idxs]

        return jnp.stack([prey_out, predator_out], axis=1)

    def loss(self, theta: jnp.ndarray) -> float:
        """
        Berechnet den MSE zwischen Simulation und Beobachtungsdaten.
        
        Args:
            theta: Parameter [α, β, δ, γ]
        
        Returns:
            Skalarer Verlustwert
        """
        sim = self.simulate(theta)
        prey_sim = sim[:10, 0]
        predator_sim = sim[:10, 1]
        mse_pre = jnp.mean((prey_sim - self.prey_obs) ** 2)
        mse_pred = jnp.mean((predator_sim - self.predator_obs) ** 2)
        return 0.5 * (mse_pre + mse_pred)

    def gradient(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Berechnet den Gradienten der Loss-Funktion bezüglich theta.
        
        Args:
            theta: Parameter [α, β, δ, γ]
        
        Returns:
            Gradient als jnp.ndarray
        """
        return grad(self.loss)(theta)
