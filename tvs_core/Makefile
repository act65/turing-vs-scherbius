build:
	maturin build --release
	pip install target/wheels/* --upgrade --force-reinstall

develop:
	cargo test
	maturin develop
	pytest
