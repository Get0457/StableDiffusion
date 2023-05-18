using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class UNet
    {

        public static Tensor<float> GenerateLatentSample(StableDiffusionConfig config, int seed, float initNoiseSigma)
        {
            return GenerateLatentSample(config.Height, config.Width, seed, initNoiseSigma);
        }
        public static Tensor<float> GenerateLatentSample(int height, int width, int seed, float initNoiseSigma)
        {
            var random = new Random(seed);
            var batchSize = 1;
            var channels = 4;
            var latents = new DenseTensor<float>(new[] { batchSize, channels, height / 8, width / 8 });

            for (int i = 0; i < latents.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number

                // add noise to latents with * scheduler.init_noise_sigma
                // generate randoms that are negative and positive
                latents.SetValue(i, (float)standardNormalRand * initNoiseSigma);
            }

            return latents;

        }

        private static void PerformGuidance(Tensor<float> noisePred, Tensor<float> noisePredText, double guidanceScale)
        {
            for (int i = 0; i < noisePred.Length; i++)
            {
                var np = noisePred.GetValue(i);
                var npt = noisePredText.GetValue(i);
                noisePred.SetValue(i, np + (float)guidanceScale * (npt - np));
            }
        }

        public static SixLabors.ImageSharp.Image Inference(string prompt, StableDiffusionConfig config)
        {
            // Preprocess text
            var textEmbeddings = TextProcessing.PreprocessText(prompt, config);
            
            var scheduler = new LMSDiscreteScheduler();
            //var scheduler = new EulerAncestralDiscreteScheduler();
            var timesteps = scheduler.SetTimesteps(config.NumInferenceSteps);
            //  If you use the same seed, you will get the same image result.
            //var seed = new Random().Next();
            var seed = 329922609;
            Console.WriteLine($"Seed generated: {seed}");
            // create latent tensor

            var latents = GenerateLatentSample(config, seed, scheduler.InitNoiseSigma);

            var sessionOptions = config.GetSessionOptionsForEp();
            // Create Inference Session
            using var unetSession = new InferenceSession(config.UnetOnnxPath, sessionOptions);
            
            GC.Collect();

            for (int t = 0; t < timesteps.Length; t++)
            {
                // torch.cat([latents] * 2)
                Tensor<float> latentModelInput = new HStackTensor<float>(latents, latents);

                // latent_model_input = scheduler.scale_model_input(latent_model_input, timestep = t)
                latentModelInput = scheduler.ScaleInput(latentModelInput, timesteps[t]);

                Console.WriteLine($"Running Step {t+1}/{timesteps.Length}");

                // Run Inference
                using var output = unetSession.Run(new List<NamedOnnxValue> {
                    NamedOnnxValue.CreateFromTensor("encoder_hidden_states", textEmbeddings),
                    NamedOnnxValue.CreateFromTensor("sample", latentModelInput),
                    NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<float>(new float[] { timesteps[t] }, new int[] { 1 }))
                });
                var outputTensor = (DenseTensor<float>)output.First().Value;

                // Split tensors from 2,4,64,64 to 1,4,64,64
                var splitTensors = TensorHelper.SplitTensor(outputTensor);
                var noisePred = splitTensors.Item1;
                var noisePredText = splitTensors.Item2;

                // Perform guidance
                PerformGuidance(noisePred, noisePredText, config.GuidanceScale);

                // LMS Scheduler Step
                scheduler.Step(noisePred, timesteps[t], latents);
                Console.WriteLine($"Finished Running Step {t + 1}");
                GC.Collect();
            }

            // Scale and decode the image latents with vae.
            // latents = 1 / 0.18215 * latents
            const float multiplier = (1.0f / 0.18215f);
            for (var i = 0; i < latents.Length; i++) {
                latents.SetValue(i, latents.GetValue(i) * multiplier);
            }
            var decoderInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("latent_sample", latents) };

            // Decode image
            var imageResultTensor = VaeDecoder.Decoder(decoderInput, config.VaeDecoderOnnxPath);

            // TODO: Fix safety checker model
            //var isSafe = SafetyChecker.IsSafe(imageResultTensor);

            ////if (isSafe == 1)
            //{ 
            var image = VaeDecoder.ConvertToImage(imageResultTensor, config);
            return image;
            //}

        }

    }
}
