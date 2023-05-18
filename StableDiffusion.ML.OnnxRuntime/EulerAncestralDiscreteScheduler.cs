using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class EulerAncestralDiscreteScheduler : SchedulerBase
    {
        private readonly string _predictionType;
        public override float InitNoiseSigma { get; set; }
        public int num_inference_steps;
        public override List<int> Timesteps { get; set; }
        public override Tensor<float> Sigmas { get; set; }

        public EulerAncestralDiscreteScheduler(
            int num_train_timesteps = 1000,
            float beta_start = 0.00085f,
            float beta_end = 0.012f,
            string beta_schedule = "scaled_linear",
            List<float> trained_betas = null,
            string prediction_type = "epsilon"
        ) : base(num_train_timesteps)
        {
            var alphas = new List<float>();
            var betas = new List<float>();
            _predictionType = prediction_type;

            if (trained_betas != null)
            {
                betas = trained_betas;
            }
            else if (beta_schedule == "linear")
            {
                betas = Enumerable.Range(0, num_train_timesteps).Select(i => beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1)).ToList();
            }
            else if (beta_schedule == "scaled_linear")
            {
                var start = (float)Math.Sqrt(beta_start);
                var end = (float)Math.Sqrt(beta_end);
                betas = np.linspace(start, end, num_train_timesteps).ToArray<float>().Select(x => x * x).ToList();

            }
            else
            {
                throw new Exception("beta_schedule must be one of 'linear' or 'scaled_linear'");
            }

            alphas = betas.Select(beta => 1 - beta).ToList();

            this._alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();
            // Create sigmas as a list and reverse it
            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

            // standard deviation of the initial noise distrubution
            this.InitNoiseSigma = (float)sigmas.Max();
        }

        public override int[] SetTimesteps(int num_inference_steps)
        {
            double start = 0;
            double stop = _numTrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            this.Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            this.InitNoiseSigma = (float)sigmas.Max();
            this.Sigmas = new DenseTensor<float>(sigmas.Count());
            for (int i = 0; i < sigmas.Count(); i++)
            {
                this.Sigmas[i] = (float)sigmas[i];
            }
            return this.Timesteps.ToArray();

        }

        public override void Step(Tensor<float> modelOutput,
               int timestep,
               Tensor<float> sample,
               int order = 4)
        {

            if (!this.is_scale_input_called)
            {
                Console.WriteLine(
                    "The `scale_model_input` function should be called before `step` to ensure correct denoising. " +
                    "See `StableDiffusionPipeline` for a usage example."
                );
            }


            int stepIndex = this.Timesteps.IndexOf((int)timestep);
            var sigma = this.Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            Tensor<float> predOriginalSample = _predictionType switch
            {
                "epsilon" =>
                    //  pred_original_sample = sample - sigma * model_output
                    TensorHelper.SubtractTensors(
                        sample,
                        TensorHelper.MultipleTensorByFloat(modelOutput, sigma)
                    ),
                "v_prediction" or "sample" => throw new NotImplementedException(
                    $"prediction_type not implemented yet: {_predictionType}"
                ),
                _ => throw new ArgumentException(
                    $"prediction_type given as {this._predictionType} must be one of `epsilon`, or `v_prediction`"
                )
            };

            float sigmaFrom = this.Sigmas[stepIndex];
            float sigmaTo = this.Sigmas[stepIndex + 1];

            var sigmaFromLessSigmaTo = (MathF.Pow(sigmaFrom, 2) - MathF.Pow(sigmaTo, 2));
            var sigmaUpResult = (MathF.Pow(sigmaTo, 2) * sigmaFromLessSigmaTo) / MathF.Pow(sigmaFrom, 2);
            var sigmaUp = sigmaUpResult < 0 ? -MathF.Pow(MathF.Abs(sigmaUpResult), 0.5f) : MathF.Pow(sigmaUpResult, 0.5f);

            var sigmaDownResult = (MathF.Pow(sigmaTo, 2) - MathF.Pow(sigmaUp, 2));
            var sigmaDown = sigmaDownResult < 0 ? -MathF.Pow(MathF.Abs(sigmaDownResult), 0.5f) : MathF.Pow(sigmaDownResult, 0.5f);

            // 2. Convert to an ODE derivative

            float dt = sigmaDown - sigma;

            var random = new Random();
            // prevSample = sample + derivative * dt;
            var prevSample = new PipelineTensor<float>(sample)
                .InPlaceContinueWith((i, x) => x +
                    (
                        // derivative
                        (
                            // sampleMinusPredOriginalSample
                            x - predOriginalSample.GetValue(i)
                        ) / sigma
                    ) * dt

                    + (
                        // noiseSigmaUpProduct
                        GetNormalRandomValue(random) * sigmaUp
                    )
                )
                ;
            prevSample.EvaluateAndWriteTo(sample);
        }

        static float GetNormalRandomValue(Random random)
        {
            // Generate a random number from a normal distribution with mean 0 and variance 1
            var u1 = random.NextDouble(); // Uniform(0,1) random number
            var u2 = random.NextDouble(); // Uniform(0,1) random number
            var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
            var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
            var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number
            return (float)standardNormalRand;
        }
    }
}
