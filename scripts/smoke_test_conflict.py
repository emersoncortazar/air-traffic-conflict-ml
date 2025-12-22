from atcml.sim.dynamics import aircraft_state
from atcml.sim.conflict import is_loss_of_separation, horizontal_separation_nm, vertical_separation_ft


def main():
    a = aircraft_state(x_nm=0.0, y_nm=0.0, alt_ft=30000.0, speed_kts=0.0, heading_deg=0.0)
    b = aircraft_state(x_nm=3.0, y_nm=4.0, alt_ft=29500.0, speed_kts=0.0, heading_deg=0.0)

    # horizontal: sqrt(3^2 + 4^2) = 5.0 NM (exactly)
    # vertical: 500 ft
    print("Horizontal sep (NM):", horizontal_separation_nm(a, b))
    print("Vertical sep (ft):", vertical_separation_ft(a, b))
    print("Loss of separation?:", is_loss_of_separation(a, b))

    print("\nExpected:")
    print("- Horizontal = 5.0 (NOT less than 5.0), Vertical = 500")
    print("- So loss of separation should be False with our strict < thresholds.")

if __name__ == "__main__":
    main()
