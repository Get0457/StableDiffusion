using StableDiffusion.ML.OnnxRuntime;

namespace StableDiffusion
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.Write("Image Geneartor\nTell me what image you want\n> ");
            var prompt = Console.ReadLine();


            //test how long this takes to execute
            var watch = System.Diagnostics.Stopwatch.StartNew();

            const string ModelPath = @"path/to/model";

            var config = new StableDiffusionConfig
            {
                // Number of denoising steps
                NumInferenceSteps = 15,
                // Scale for classifier-free guidance
                GuidanceScale = 7.5,
                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.DirectML,
                // Set GPU Device ID.
                DeviceId = 1,
                //Width = 512 / 2,
                //Height = 512 / 2,
                // Update paths to your models
                TextEncoderOnnxPath = $@"{ModelPath}\text_encoder\model.onnx",
                UnetOnnxPath = $@"{ModelPath}\unet\model.onnx",
                VaeDecoderOnnxPath = $@"{ModelPath}\vae_decoder\model.onnx",
                SafetyModelPath = $@"{ModelPath}\safety_checker\model.onnx",
            };

            // Inference Stable Diff
            var image = UNet.Inference(prompt, config);

            // If image failed or was unsafe it will return null.
            if (image == null)
            {
                Console.WriteLine("Unable to create image, please try again.");
            }
            // Stop the timer
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Time taken: " + elapsedMs + "ms");

        }

    }
}