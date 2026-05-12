import os
import shutil
import random
import glob
from collections import Counter

class DatasetHandler:
    def __init__(self, raw_images_dir, raw_labels_dir, output_dir, split_ratios=(0.7, 0.2, 0.1)):
        """
        Initializes the DatasetHandler.
        split_ratios: (train, val, test)
        """
        self.raw_images_dir = raw_images_dir
        self.raw_labels_dir = raw_labels_dir
        self.output_dir = output_dir
        self.split_ratios = split_ratios
        self.classes = [
            "normal_person",
            "wheelchair_user",
            "blind_person",
            "crutch_user"
        ]

    def _create_dirs(self):
        """Creates the train, val, test directory structure for YOLO."""
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'labels', split), exist_ok=True)

    def verify_and_get_valid_pairs(self):
        """Verifies missing labels/images and returns valid pairs."""
        valid_pairs = []
        missing_labels = 0

        if not os.path.exists(self.raw_images_dir) or not os.path.exists(self.raw_labels_dir):
            print("Raw images or labels directory not found. Assuming dataset is already prepared or empty.")
            return valid_pairs

        image_files = glob.glob(os.path.join(self.raw_images_dir, '*.*'))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        for img_path in image_files:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.raw_labels_dir, f"{filename}.txt")

            if os.path.exists(label_path):
                valid_pairs.append((img_path, label_path))
            else:
                missing_labels += 1

        print(f"Total valid image-label pairs found: {len(valid_pairs)}")
        if missing_labels > 0:
            print(f"Warning: Found {missing_labels} images without corresponding label files.")

        return valid_pairs

    def process_and_split(self):
        """Splits the valid pairs into train, val, and test sets and copies them."""
        valid_pairs = self.verify_and_get_valid_pairs()
        if not valid_pairs:
            print("No valid pairs to process.")
            return

        self._create_dirs()

        # Shuffle randomly
        random.shuffle(valid_pairs)

        total = len(valid_pairs)
        train_end = int(total * self.split_ratios[0])
        val_end = train_end + int(total * self.split_ratios[1])

        splits = {
            'train': valid_pairs[:train_end],
            'val': valid_pairs[train_end:val_end],
            'test': valid_pairs[val_end:]
        }

        for split_name, pairs in splits.items():
            print(f"Processing {split_name} split: {len(pairs)} files...")
            for img_path, label_path in pairs:
                # Copy image
                shutil.copy(img_path, os.path.join(self.output_dir, 'images', split_name, os.path.basename(img_path)))
                # Copy label
                shutil.copy(label_path, os.path.join(self.output_dir, 'labels', split_name, os.path.basename(label_path)))

        print("Dataset split complete.")

    def display_statistics(self):
        """Displays statistics on the number of instances for each class in the output dataset."""
        label_files = glob.glob(os.path.join(self.output_dir, 'labels', '*', '*.txt'))
        class_counts = Counter()

        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

        print("\n--- Dataset Statistics ---")
        for class_id in range(len(self.classes)):
            count = class_counts.get(class_id, 0)
            print(f"Class {class_id} ({self.classes[class_id]}): {count} instances")
        print("--------------------------\n")

if __name__ == "__main__":
    # Example usage:
    # Assuming raw data is in 'dataset/raw/images' and 'dataset/raw/labels'
    handler = DatasetHandler(
        raw_images_dir='dataset/raw/images',
        raw_labels_dir='dataset/raw/labels',
        output_dir='dataset'
    )
    handler.process_and_split()
    handler.display_statistics()
