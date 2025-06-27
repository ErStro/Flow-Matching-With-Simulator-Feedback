import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sbibm.tasks import get_task

from evaluate_baseline_with_sim import evaluate_baseline_with_sim
from lotka_volterra_loss import LotkaVolterraLoss


if __name__ == "__main__":
                          
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_file_path = os.path.abspath(
        os.path.join(script_dir, "..", "baseline_net", "flow_matching_model.py")
    )
    model_path = os.path.abspath(
        os.path.join(script_dir, "..", "baseline_net", "baseline_model.pt")
    )

                                    
    task = get_task("lotka_volterra")
    x_obs = task.get_observation(8)[0].numpy()         
    x_obs_jnp = jnp.array(x_obs)

                                                                   
    mse, theta_hat, x_sim, loss, gradient = evaluate_baseline_with_sim(
        model_path, model_file_path, x_obs
    )

                            
    print("----------------------------------------------------")
    print("🎯 Inferenzierte Parameter (theta_hat):")
    print(theta_hat)
    print("\n📉 MSE zwischen x_obs und x_sim (klassisch):")
    print(float(mse))
    print("\n🧮 Loss (LotkaVolterraLoss-Klasse):")
    print(float(loss))
    print("\n🔁 Gradient ∇Loss(theta_hat):")
    print(gradient)
    print("----------------------------------------------------")


                                                
                                                                                   
    x_obs_pred = x_obs_jnp[:10]
    x_obs_prey = x_obs_jnp[10:]
    x_obs_split = jnp.stack([x_obs_prey, x_obs_pred], axis=-1)
    x_sim_cut = x_sim[:10]                                            

                                     
    print("\n🧾 Vergleich Beobachtung vs. Simulation:")
    print(f"{'t':>3} | {'x_obs_prey':>12} | {'x_sim_prey':>12} | {'x_obs_pred':>13} | {'x_sim_pred':>13}")
    print("-" * 64)
    for i in range(10):
        print(f"{i:>3} | {x_obs_split[i,0]:>12.5f} | {x_sim_cut[i,0]:>12.5f} | {x_obs_split[i,1]:>13.5f} | {x_sim_cut[i,1]:>13.5f}")

                      
    plt.plot(x_obs[::2], "--", label="Obs Prey")
    plt.plot(x_obs[1::2], "--", label="Obs Predator")
    plt.plot(x_sim[:, 0], label="Sim Prey (θ̂)")
    plt.plot(x_sim[:, 1], label="Sim Predator (θ̂)")
    plt.xlabel("Zeit")
    plt.ylabel("Population")
    plt.title("BaselineNet: Simulation vs Observation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
