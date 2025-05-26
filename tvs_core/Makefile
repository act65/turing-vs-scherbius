build:
	cargo test

	maturin develop
	maturin build --release

	pip install target/wheels/turing_vs_scherbius-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl --force-reinstall

	pytest