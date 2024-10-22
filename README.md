✨ Text-to-Image Generation using GANs

Turn words into stunning images using GANs! 🎉

🚀 Features

	•	🎨 Generate images from random noise using GANs.
	•	🧠 Built with PyTorch for easy model development.
	•	🔥 Google Colab support for GPU-based training and inference.
	•	🗂 COCO dataset to map captions to images.
	•	💾 Pre-trained model support to skip training and directly generate images.


💻 Technologies Used

	•	Python 🐍
	•	PyTorch: For GAN architecture
	•	Google Colab: GPU-powered training 🚀
	•	COCO Dataset: Real-world image-caption pairs 🖼
	•	Matplotlib: Image visualization 📊
	•	NLTK: For text tokenization ✏️

🌐 How to Use the Project

Step 1️⃣: Open Colab Notebook

Head over to Google Colab and create a new notebook.

Step 2️⃣: Set GPU Runtime

	•	Click on Runtime > Change runtime type.
	•	Set Hardware accelerator to GPU. ⚡

Step 3️⃣: Copy and Paste the Code

	•	Paste the full code provided into your Colab notebook.

Step 4️⃣: Download the Dataset

!mkdir coco_data
!wget http://images.cocodataset.org/zips/train2017.zip -O coco_data/train2017.zip
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O coco_data/annotations.zip
!unzip coco_data/train2017.zip -d coco_data/
!unzip coco_data/annotations.zip -d coco_data/

Step 5️⃣: Run the Notebook

Click Runtime > Run all to execute all the cells. 🏃‍♂️

🔄 Training the Model

During training, you will see logs showing discriminator and generator loss values, helping you track model performance:

Epoch [1/5], Step [101/9247], D Loss: 0.1281, G Loss: 4.7560

🖼 Generate Images from Random Noise

Once the model is trained, use this snippet to generate a new image from latent vectors:

def generate_image():
    z = torch.randn(1, 100).to(device)
    with torch.no_grad():
        img = generator(z).cpu().squeeze(0)
    return img

img = generate_image()
plt.imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5)
plt.axis('off')
plt.show()

🎉 Output:

📋 Installation Requirements

You can run everything on Google Colab. If running locally, use the following command to install dependencies:

pip install torch torchvision nltk pycocotools matplotlib Pillow

📈 Future Improvements

	•	🧑‍🎨 Add text-conditioning: Generate images directly from user-provided text.
	•	🏞 Use advanced GAN models like StyleGAN for high-quality outputs.
	•	🕸 Deploy a web interface to allow real-time text-to-image generation.

🤝 Contributions

We welcome contributions! Please fork the repository and submit pull requests for new features or improvements.

📝 License

This project is licensed under the MIT License. Feel free to use it for personal or academic purposes.

💡 Acknowledgments

	•	Google Colab for free access to GPUs 🎉
	•	COCO Dataset creators for their amazing dataset 🖼
	•	PyTorch for the GAN framework 🧠
