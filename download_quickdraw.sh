#!/usr/bin/env bash
BASE_URL="https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
DEST="data"

FILES=(
  "The%20Eiffel%20Tower.npy" "The%20Great%20Wall%20of%20China.npy" "The%20Mona%20Lisa.npy"
  "aircraft%20carrier.npy" "airplane.npy" "alarm%20clock.npy" "ambulance.npy" "angel.npy" "animal%20migration.npy"
  "ant.npy" "anvil.npy" "apple.npy" "arm.npy" "asparagus.npy" "axe.npy" "backpack.npy" "banana.npy" "bandage.npy"
  "barn.npy" "baseball%20bat.npy" "baseball.npy" "basket.npy" "basketball.npy" "bat.npy" "bathtub.npy" "beach.npy"
  "bear.npy" "beard.npy" "bed.npy" "bee.npy" "belt.npy" "bench.npy" "bicycle.npy" "binoculars.npy" "bird.npy"
  "birthday%20cake.npy" "blackberry.npy" "blueberry.npy" "book.npy" "boomerang.npy" "bottlecap.npy" "bowtie.npy"
  "bracelet.npy" "brain.npy" "bread.npy" "bridge.npy" "broccoli.npy" "broom.npy" "bucket.npy" "bulldozer.npy"
)

for f in "${FILES[@]}"; do
  echo "Downloading $f..."
  wget -q --show-progress -P "$DEST" "$BASE_URL/$f"
done