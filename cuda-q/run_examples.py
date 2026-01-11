"""Main script to run VQA examples."""
import sys
import argparse


def main():
    """Run VQA examples based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run VQA examples for Ising model or H2 molecule'
    )

    parser.add_argument(
        'model',
        choices=['ising', 'h2', 'both', 'test'],
        help='Which model to run (ising, h2, both, or test)'
    )

    parser.add_argument(
        '--qubits',
        type=int,
        default=4,
        help='Number of qubits for Ising model (default: 4)'
    )

    parser.add_argument(
        '--J',
        type=float,
        default=1.0,
        help='Coupling strength for Ising model (default: 1.0)'
    )

    parser.add_argument(
        '--h',
        type=float,
        default=0.5,
        help='Transverse field strength for Ising model (default: 0.5)'
    )

    parser.add_argument(
        '--distance',
        type=float,
        default=0.74,
        help='Bond distance for H2 molecule in Angstroms (default: 0.74)'
    )

    parser.add_argument(
        '--depth',
        type=int,
        default=2,
        help='Ansatz depth (default: 2)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='COBYLA',
        choices=['COBYLA', 'Nelder-Mead', 'Powell'],
        help='Optimization method (default: COBYLA)'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=200,
        help='Maximum iterations (default: 200)'
    )

    args = parser.parse_args()

    if args.model == 'test':
        # Run tests
        print("Running tests...\n")
        sys.path.append('tests')
        import run_all_tests
        run_all_tests.main()

    elif args.model == 'ising' or args.model == 'both':
        # Run Ising model
        print("\n" + "=" * 70)
        print("Running Ising Model VQA")
        print("=" * 70)
        print(f"Parameters: qubits={args.qubits}, J={args.J}, h={args.h}")
        print(f"Ansatz depth: {args.depth}")
        print(f"Optimization: {args.method}, max_iterations={args.iterations}")
        print("=" * 70 + "\n")

        sys.path.append('examples')
        from ising_vqa import run_ising_vqa

        energy, params = run_ising_vqa(
            num_qubits=args.qubits,
            J=args.J,
            h=args.h,
            depth=args.depth,
            periodic=True,
            method=args.method,
            max_iterations=args.iterations
        )

    if args.model == 'h2' or args.model == 'both':
        # Run H2 molecule
        print("\n" + "=" * 70)
        print("Running H2 Molecule VQA")
        print("=" * 70)
        print(f"Parameters: bond_distance={args.distance} Å")
        print(f"Optimization: {args.method}, max_iterations={args.iterations}")
        print("=" * 70 + "\n")

        sys.path.append('examples')
        from h2_vqa import run_h2_vqa

        energy, params = run_h2_vqa(
            bond_distance=args.distance,
            ansatz_type='simplified',
            method=args.method,
            max_iterations=args.iterations
        )

    print("\n" + "=" * 70)
    print("Execution complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
