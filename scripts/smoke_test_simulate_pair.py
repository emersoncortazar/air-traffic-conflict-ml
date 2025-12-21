from src.sim.encounter_generators import make_head_on
from src.sim.simulate import simulate_pair

def main():
    a, b = make_head_on(separation_nm=20.0, speed_kts=360.0, alt_ft=30000.0)
    rows = simulate_pair(a, b, dt_s=10.0, horizon_s=50.0)

    print("Number of rows:", len(rows))
    print("First row:", rows[0])
    print("Last row:", rows[-1])

    print("\nExpected:")
    print("- dt=10, horizon=50 => times 0,10,20,30,40,50 => 6 rows")
    print("- a_x_nm should increase each step")
    print("- b_x_nm should decrease each step")

if __name__ == "__main__":
    main()
