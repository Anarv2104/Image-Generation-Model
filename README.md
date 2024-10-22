✨ Text-to-Image Generation using GANs

Transform random noise into visually stunning images! 🎉

🚀 Features

	•	🎨 Generate images from random noise using GANs.
	•	🧠 PyTorch for easy deep learning development.
	•	🔥 Google Colab support for GPU-powered training and inference.
	•	🗂 COCO dataset to map captions to images.
	•	💾 Pre-trained model support for fast image generation.

💻 Technologies Used

	•	Python 🐍
	•	PyTorch: For GAN development
	•	Google Colab: GPU-powered training environment 🚀
	•	COCO Dataset: Real-world image-caption pairs 🖼
	•	Matplotlib: Visualizing generated images 📊
	•	NLTK: Tokenizing text captions ✏️

🌐 How to Use the Project

Step 1️⃣: Open the Colab Notebook

Head over to Google Colab and create a new notebook.

Step 2️⃣: Set Runtime to GPU

	•	Click Runtime > Change runtime type.
	•	Set Hardware accelerator to GPU. ⚡

Step 3️⃣: Download the Dataset

Run the following commands in a Colab cell to download the MS-COCO dataset:

	!mkdir coco_data
	!wget http://images.cocodataset.org/zips/train2017.zip -O coco_data/train2017.zip
	!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O coco_data/annotations.zip
	!unzip coco_data/train2017.zip -d coco_data/
	!unzip coco_data/annotations.zip -d coco_data/

Step 4️⃣: Run the Notebook

	•	Click Runtime > Run all to execute the entire code.

🔄 Training the Model

The training loop will print the discriminator and generator losses every 100 steps. Example:

	Epoch [1/5], Step [101/9247], D Loss: 0.1281, G Loss: 4.7560

🖼 Generate Images from Random Noise

Use the following code snippet to generate a new image:

		def generate_image():
		    z = torch.randn(1, 100).to(device)  # Generate random noise
		    with torch.no_grad():
		        img = generator(z).cpu().squeeze(0)  # Generate image
		    return img
		
		img = generate_image()
		plt.imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Rescale for display
		plt.axis('off')
		plt.show()

🎉 Expected Output:

📋 Installation Requirements

If you want to run the project locally, install the dependencies using:

pip install torch torchvision nltk pycocotools matplotlib Pillow

📈 Future Improvements

	•	🧑‍🎨 Text-conditioned GANs: Generate images based on user input text.
	•	🏞 Advanced GAN architectures like StyleGAN for improved results.
	•	🕸 Deploy a web interface for real-time image generation.

🤝 Contributions

We welcome contributions! Fork the repository and submit pull requests to add new features or enhance existing ones.

💡 Acknowledgments

	•	Google Colab for free GPU access 🎉
	•	MS-COCO dataset for high-quality image-caption pairs 🖼
	•	PyTorch for the GAN framework 🧠
