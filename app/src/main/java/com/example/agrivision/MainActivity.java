package com.example.agrivision;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_REQUEST_CODE = 3;
    private static final int GALLERY_REQUEST_CODE = 1;
    private static final int CAMERA_PERMISSION_CODE = 100;
    private static final int IMAGE_SIZE = 224;

    private Button cameraButton, galleryButton;
    private ImageView imageView;
    private TextView resultTextView;
    private TextView subtitleTextView;
    private ProgressBar progressBar;
    private Interpreter interpreter;
    private Executor executor;

    // Pre-allocate the buffer for image processing
    private ByteBuffer imageBuffer;
    private int[] pixels;
    private float[][] outputBuffer;
    private String[] labels = {"Brown Spot", "Healthy", "Rice Blast", "Bacterial Leaf Blight", "Narrow Brown Spot", "Sheath Blight"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        // Initialize UI components
        initializeViews();

        // Initialize TensorFlow Lite interpreter
        try {
            interpreter = new Interpreter(loadModelFile(), null);

            // Pre-allocate buffers for better performance
            imageBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3);
            imageBuffer.order(ByteOrder.nativeOrder());
            pixels = new int[IMAGE_SIZE * IMAGE_SIZE];
            outputBuffer = new float[1][6];

            // Create a single thread executor for background tasks
            executor = Executors.newSingleThreadExecutor();
        } catch (IOException e) {
            Log.e("AgriVision", "Failed to load model", e);
        }

        // Set click listeners
        setupClickListeners();
    }

    private void initializeViews() {
        cameraButton = findViewById(R.id.btnCamera);
        galleryButton = findViewById(R.id.btnGallery);
        imageView = findViewById(R.id.ImageView);
        resultTextView = findViewById(R.id.titleText);
        subtitleTextView = findViewById(R.id.subtitleText);
        progressBar = findViewById(R.id.progressBar);
    }

    private void setupClickListeners() {
        cameraButton.setOnClickListener(view -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
            }
        });

        galleryButton.setOnClickListener(view -> {
            Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(galleryIntent, GALLERY_REQUEST_CODE);
        });
    }

    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, CAMERA_REQUEST_CODE);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        try (AssetFileDescriptor fileDescriptor = getAssets().openFd("model.tflite");
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {

            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            try {
                Bitmap image = null;

                if (requestCode == CAMERA_REQUEST_CODE) {
                    image = (Bitmap) data.getExtras().get("data");
                } else if (requestCode == GALLERY_REQUEST_CODE) {
                    Uri imageUri = data.getData();
                    if (imageUri != null) {
                        image = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                    }
                }

                if (image != null) {
                    // Display original image
                    imageView.setImageBitmap(image);

                    // Start classification
                    startImageClassification(image);
                }
            } catch (IOException e) {
                Log.e("AgriVision", "Error processing image", e);
            }
        }
    }

    private void startImageClassification(final Bitmap originalImage) {
        // Show loading state
        showLoadingState();

        // Process on background thread
        executor.execute(() -> {
            // Resize image for the model
            Bitmap resizedImage = Bitmap.createScaledBitmap(originalImage, IMAGE_SIZE, IMAGE_SIZE, false);

            // Classify the image
            String result = classifyImage(resizedImage);

            // Update UI on main thread
            runOnUiThread(() -> {
                showResultState(result);
            });
        });
    }

    private void showLoadingState() {
        resultTextView.setVisibility(View.GONE);
        progressBar.setVisibility(View.VISIBLE);
        subtitleTextView.setText("Analyzing image...");
    }

    private void showResultState(String result) {
        progressBar.setVisibility(View.GONE);
        resultTextView.setText(result);
        resultTextView.setVisibility(View.VISIBLE);
        subtitleTextView.setText("Capture or Upload another image to analyze");
    }

    private String classifyImage(Bitmap image) {
        // Clear the buffer before reuse
        imageBuffer.rewind();

        // Get image pixels
        image.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

        // Convert the image to floating point
        for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
            int pixel = pixels[i];
            imageBuffer.putFloat(((pixel >> 16) & 0xFF)); // Red
            imageBuffer.putFloat(((pixel >> 8) & 0xFF));  // Green
            imageBuffer.putFloat((pixel & 0xFF));         // Blue
        }

        // Run inference
        interpreter.run(imageBuffer, outputBuffer);

        // Find class with highest confidence
        int maxIndex = 0;
        for (int i = 1; i < outputBuffer[0].length; i++) {
            if (outputBuffer[0][i] > outputBuffer[0][maxIndex]) {
                maxIndex = i;
            }
        }

        float confidence = outputBuffer[0][maxIndex] * 100;
        return String.format("Prediction: %s (%.1f%%)", labels[maxIndex], confidence);
    }

    @Override
    protected void onDestroy() {
        if (interpreter != null) {
            interpreter.close();
        }
        super.onDestroy();
    }
}