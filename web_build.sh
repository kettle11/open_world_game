RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 -Clink-arg=--max-memory=4294967296' \
  cargo build --target wasm32-unknown-unknown -Z build-std=std,panic_abort --release
cp target/wasm32-unknown-unknown/release/open_world_game.wasm web_build/wasm.wasm
wasm-opt web_build/wasm.wasm -o web_build/wasm.wasm -O