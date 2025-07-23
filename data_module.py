from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        # instance_data_root,
        instance_prompt,
        class_prompt,
        train_text_encoder_ti,
        token_abstraction_dict=None,  # token mapping for textual inversion
        # class_data_root=None,
        class_num=None,
        # size=512,
        repeats=1,
        center_crop=False,
    ):
        # self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti
  
        dataset_name = "raresense/Master_Jewelry_HD"

        # Load the dataset directly
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. Please install the datasets library: `pip install datasets`."
            )

        # Load the dataset from Hugging Face Hub
        dataset = load_dataset(dataset_name)
        # instance_images = dataset["train"]["target"]
        instance_images = dataset["train"]["source"]#image-1 on left side

        source_images = dataset["train"]["target"] #image-2 on right side
        # Use a static caption for all images
        instance_prompts = dataset["train"]["caption"]

        mask= "mask_square.png"
        # No need for caption column; set custom_instance_prompts using the static caption
        # self.custom_instance_prompts = [instance_prompt] * len(instance_images) * repeats


        # Repeat the images based on the repeats argument
        self.instance_images = list(itertools.chain.from_iterable(itertools.repeat(img, repeats) for img in instance_images))

        # Repeat the source images based on the repeats argument
        self.source_images = list(itertools.chain.from_iterable(itertools.repeat(img, repeats) for img in source_images))

        self.custom_instance_prompts = list(itertools.chain.from_iterable(itertools.repeat(p, repeats) for p in instance_prompts))

        # Repeat the mask image for all instances
        self.mask_images = [mask] * len(self.instance_images)  
        self.pixel_values = []
        self.mask_pixel_values =[]
        # self.masked_images = []
        # train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        # train_flip = transforms.RandomHorizontalFlip(p=1.0)
        mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),   
            ]
        )
        train_transforms = transforms.Compose(
            [
                transforms.Resize((768, 576), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # Iterate over both instance and source images
        for instance_image, source_image in zip(self.instance_images, self.source_images):
            # Ensure both images are properly formatted
            instance_image = exif_transpose(instance_image)
            source_image = exif_transpose(source_image)
            
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            if not source_image.mode == "RGB":
                source_image = source_image.convert("RGB")

          
            instance_tensor = train_transforms(instance_image)  # Apply transformations to instance_image
            source_tensor = train_transforms(source_image)     # Apply transformations to source_image

            # concatenated_image = torch.cat([instance_tensor, source_tensor], dim=2)  # Concatenate along width
            image = torch.cat([instance_tensor, source_tensor], dim=2)
            self.pixel_values.append(image)


        for mask_path in self.mask_images:
            # Load the mask as a PIL.Image
            mask = Image.open(mask_path).convert("L").resize((1152,768))  # Ensure it is in RGB mode

            # Apply the transformation to convert to grayscale and tensor
            mask= mask_transform(mask)

            # Append the transformed mask to mask_pixel_values
            self.mask_pixel_values.append(mask)

        for image, mask in zip(self.pixel_values, self.mask_pixel_values):
            # Ensure mask is compatible with the image dimensions
            if mask.shape[1:] != image.shape[1:]:
                mask = transforms.Resize(image.shape[1:])(mask)

            # Compute masked_image
            # masked_image = image * (1 - mask)
            # self.masked_images.append(masked_image)

        # Update the instance count and dataset length
        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        # Fetch the concatenated image from pixel_values
        image = self.pixel_values[index % self.num_instance_images]
        mask = self.mask_pixel_values[index % self.num_instance_images]
        caption = self.custom_instance_prompts[index % self.num_instance_images]

        example["instance_images"] = image  # Use the concatenated image as the main input
        example["instance_masks"] = mask
        # Handle instance prompt
        example["instance_prompt"] = caption

        return example

class DreamBoothDataset2(Dataset):
    """
    A dataset to prepare only target images with masks from the dataset for fine-tuning.
    No image concatenation is performed.
    """

    def __init__(
        self,
        dataset_name,
        target_column_name,
        mask_column_name,
        caption_column_name,
        static_caption=None,
        size=(768, 1024),
        repeats=1,
    ):
        self.dataset_name = dataset_name
        self.target_column_name = target_column_name
        self.mask_column_name = mask_column_name
        self.caption_column_name = caption_column_name
        self.static_caption = static_caption
        self.size = size

        # Load the dataset directly
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. Please install the datasets library: `pip install datasets`."
            )

        # Load the dataset from Hugging Face Hub
        dataset = load_dataset(self.dataset_name)
        self._length = len(self.dataset["train"])
        sample = self.dataset["train"][index]
        target_images = sample[self.target_column_name]
        mask_images = sample[self.mask_column_name]
        instance_prompts = sample[self.caption_column_name]

        # # Only use target images (no source images)
        # target_images = dataset["train"]["target"]
        
        # # Get masks from dataset instead of local path
        # mask_images = dataset["train"]["mask"]  # Assuming masks are in the dataset
        
        # # Use captions from dataset
        # instance_prompts = dataset["train"]["prompt"]

        # Repeat the images based on the repeats argument
        self.target_images = list(itertools.chain.from_iterable(itertools.repeat(img, repeats) for img in target_images))
        
        # Repeat the masks based on the repeats argument
        self.mask_images = list(itertools.chain.from_iterable(itertools.repeat(mask, repeats) for mask in mask_images))
        
        # Repeat the prompts based on the repeats argument
        self.custom_instance_prompts = list(itertools.chain.from_iterable(itertools.repeat(p, repeats) for p in instance_prompts))

        self.pixel_values = []
        self.mask_pixel_values = []
        
        # Transform for masks
        mask_transform = transforms.Compose([
            transforms.ToTensor(),   
        ])
        
        # Transform for target images
        train_transforms = transforms.Compose([
            transforms.Resize((1024, 768), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Process target images
        for target_image in self.target_images:
            # Ensure image is properly formatted
            target_image = exif_transpose(target_image)
            
            if not target_image.mode == "RGB":
                target_image = target_image.convert("RGB")
            
            # Apply transformations to target image only
            target_tensor = train_transforms(target_image)
            self.pixel_values.append(target_tensor)

        # Process mask images from dataset
        for mask_image in self.mask_images:
            # Ensure mask is properly formatted
            mask_image = exif_transpose(mask_image)
            
            # Convert to grayscale and resize
            mask = mask_image.convert("L").resize((768, 1024))  # Match target image size
            
            # Apply transformation to convert to tensor
            mask = mask_transform(mask)
            
            # Append the transformed mask to mask_pixel_values
            self.mask_pixel_values.append(mask)

        # Ensure mask dimensions match image dimensions
        for i, (image, mask) in enumerate(zip(self.pixel_values, self.mask_pixel_values)):
            if mask.shape[1:] != image.shape[1:]:
                mask = transforms.Resize(image.shape[1:])(mask)
                self.mask_pixel_values[i] = mask

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        # Fetch the target image (no concatenation)
        image = self.pixel_values[index % self.num_instance_images]
        mask = self.mask_pixel_values[index % self.num_instance_images]
        caption = self.custom_instance_prompts[index % self.num_instance_images]

        example["instance_images"] = image  # Single target image
        example["instance_masks"] = mask
        example["instance_prompt"] = caption

        return example

def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    mask_pixel_values = [example["instance_masks"] for example in examples]
    # masked_images= [example["masked_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask_pixel_values= torch.stack(mask_pixel_values)
    mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format).float()
    # masked_images=torch.stack(masked_images)
    # masked_images =masked_images.to(memory_format=torch.contiguous_format).float()
    
    batch = {"pixel_values": pixel_values, "mask_pixel_values": mask_pixel_values ,"prompts": prompts}
    return batch

# -------------------------------------------------------------------------------------------------------

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

if __name__ == "__main__":
    # Create dataset instance with the updated parameters:
    dataset = KontextDataset(
        dataset_name="raresense/Viton",
        target_column_name="source",
        source_column_name="target",
        caption_column_name="ai_name",
    )
    
    # Print dataset length
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    print("Loading first sample...")
    sample = dataset[0]
    
    # Print tensor shapes
    print("\nTensor shape of concatenated image:")
    print(f"Target image: {sample['target_image'].shape}")
    print(f"Mask: {sample['mask'].shape}")
    
    # Print prompts (truncated if too long)
    prompts = sample["prompts"]
    if len(prompts) > 100:
        prompts = prompts[:100] + "..."
    print(f"\nCaption: {prompts}")

    # Create and test DataLoader with custom collate function
    batch_size = 4
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print("\nTesting batch processing:")
    print(f"Getting first batch of size {batch_size}...")
    batch = next(iter(dataloader))
    
    # Print tensor shapes for batch
    print("Tensor shapes for batch:")
    print(f"Target images: {batch['target_image'].shape}")
    print(f"Masks: {batch['mask'].shape}")
    print(f"Sources: {batch['source_image'].shape}")
    print(f"Number of captions: {len(batch['prompts'])}")
    
    # Print first prompts in batch
    first_caption = batch['prompts'][0]
    if len(first_caption) > 100:
        first_caption = first_caption
    print(f"First prompts in batch: {first_caption}")

# -------------------------------------------------------------------------------------------------------
# This is a modified version of the DreamBoothDataset2 class to handle split train and test dataset for Hugging Face datasets.
class DreamBoothDataset2(Dataset):
    """
    A dataset to prepare only target images with masks from the dataset for fine-tuning.
    No image concatenation is performed.
    """

    def __init__(
        self,
        instance_prompt,
        class_prompt=None,
        train_text_encoder_ti=False,
        token_abstraction_dict=None,
        class_num=None,
        repeats=1,
        center_crop=False,
        dataset_name="raresense/Bracelets",
        split="train",                          # <---- NEW
        size=(1024, 768),                       # <---- optional, keep transforms flexible
    ):
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti
        self.size = size
        self.split = split

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install `datasets`: pip install datasets")

        ds = load_dataset(dataset_name, split=split)

        target_images = ds["target"]
        mask_images   = ds["mask"]
        instance_prompts = ds["prompt"]

        # Only repeat for training
        rep = repeats if split == "train" else 1
        self.target_images = list(itertools.chain.from_iterable(itertools.repeat(img, rep) for img in target_images))
        self.mask_images   = list(itertools.chain.from_iterable(itertools.repeat(msk, rep) for msk in mask_images))
        self.custom_instance_prompts = list(itertools.chain.from_iterable(itertools.repeat(p, rep) for p in instance_prompts))

        self.pixel_values = []
        self.mask_pixel_values = []

        mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        h, w = self.size
        train_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        for target_image in self.target_images:
            target_image = exif_transpose(target_image)
            if target_image.mode != "RGB":
                target_image = target_image.convert("RGB")
            self.pixel_values.append(train_transforms(target_image))

        for mask_image in self.mask_images:
            mask_image = exif_transpose(mask_image)
            mask = mask_image.convert("L").resize((w, h))
            self.mask_pixel_values.append(mask_transform(mask))

        # Ensure mask dims match image dims
        for i, (img, msk) in enumerate(zip(self.pixel_values, self.mask_pixel_values)):
            if msk.shape[1:] != img.shape[1:]:
                self.mask_pixel_values[i] = transforms.Resize(img.shape[1:])(msk)

        self.num_instance_images = len(self.target_images)
        self._length = self.num_instance_images

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        image  = self.pixel_values[idx % self.num_instance_images]
        mask   = self.mask_pixel_values[idx % self.num_instance_images]
        prompt = self.custom_instance_prompts[idx % self.num_instance_images]

        return {
            "instance_images": image,
            "instance_masks": mask,
            "instance_prompt": prompt,
        }