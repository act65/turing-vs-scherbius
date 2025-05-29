build:
	cargo test
	maturin build --release
	pip install target/wheels/*
	pytest