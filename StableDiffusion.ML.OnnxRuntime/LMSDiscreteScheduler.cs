﻿using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics;
using NumSharp;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class LMSDiscreteScheduler : SchedulerBase
    {
        private int _numTrainTimesteps;
        private string _predictionType;

        public override Tensor<float> Sigmas { get; set; }
        public override List<int> Timesteps { get; set; }
        public List<Tensor<float>> Derivatives;
        public override float InitNoiseSigma { get; set; }

        public LMSDiscreteScheduler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, string beta_schedule = "scaled_linear", string prediction_type = "epsilon", List<float> trained_betas = null)
        {
            _numTrainTimesteps = num_train_timesteps;
            _predictionType = prediction_type;
            Derivatives = new List<Tensor<float>>();
            Timesteps = new List<int>();

            var alphas = new List<float>();
            var betas = new List<float>();

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

        //python line 135 of scheduling_lms_discrete.py
        public double GetLmsCoefficient(int order, int t, int currentOrder)
        {
            // Compute a linear multistep coefficient.

            double LmsDerivative(double tau)
            {
                double prod = 1.0;
                for (int k = 0; k < order; k++)
                {
                    if (currentOrder == k)
                    {
                        continue;
                    }
                    prod *= (tau - this.Sigmas[t - k]) / (this.Sigmas[t - currentOrder] - this.Sigmas[t - k]);
                }
                return prod;
            }

            double integratedCoeff = Integrate.OnClosedInterval(LmsDerivative, this.Sigmas[t], this.Sigmas[t + 1], 1e-4);

            return integratedCoeff;
        }

        // Line 157 of scheduling_lms_discrete.py from HuggingFace diffusers
        public override int[] SetTimesteps(int num_inference_steps)
        {
            double start = 0;
            double stop = _numTrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            this.Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            this.Sigmas = new DenseTensor<float>(sigmas.Count());
            for (int i = 0; i < sigmas.Count(); i++)
            {
                this.Sigmas[i] = (float)sigmas[i];
            }
            return this.Timesteps.ToArray();

        }

        public override void Step(
               Tensor<float> modelOutput,
               int timestep,
               Tensor<float> sample,
               int order = 4)
        {
            int stepIndex = this.Timesteps.IndexOf(timestep);
            var sigma = this.Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            DenseTensor<float> predOriginalSample = new(modelOutput.Dimensions);

            // Create array of type float length modelOutput.length
            var modelOutPutArray = modelOutput.ToArray();
            var sampleArray = sample.ToArray();

            switch (_predictionType)
            {
                case "epsilon":
                    for (int i = 0; i < modelOutPutArray.Length; i++)
                    {
                        predOriginalSample.SetValue(i, sampleArray[i] - sigma * modelOutPutArray[i]);
                    }
                    break;
                case "v_prediction":
                    //predOriginalSample = modelOutput * ((-sigma / Math.Sqrt((Math.Pow(sigma,2) + 1))) + (sample / (Math.Pow(sigma,2) + 1)));
                    throw new Exception($"prediction_type given as {this._predictionType} not implemented yet.");
                default:
                    throw new Exception($"prediction_type given as {this._predictionType} must be one of `epsilon`, or `v_prediction`");
            }

            // 2. Convert to an ODE derivative
            var derivativeItems = new DenseTensor<float>(sample.Dimensions);

            for (int i = 0; i < modelOutPutArray.Length; i++)
            {
                //predOriginalSample = (sample - predOriginalSample) / sigma;
                derivativeItems.SetValue(i, (sampleArray[i] - predOriginalSample.GetValue(i)) / sigma);
            }

            Derivatives?.Add(derivativeItems);

            if (Derivatives?.Count > order)
            {
                // remove first element
                Derivatives?.RemoveAt(0);
            }

            // 3. compute linear multistep coefficients
            order = Math.Min(stepIndex + 1, order);
            
            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives

            // Create tensor for product of lmscoeffs and derivatives
            PipelineTensor<float> lmsDerProduct = new(sample);

            var iter = Math.Min(order, Derivatives.Count);
            var lastDerIdx = Derivatives.Count - 1;
            int m = 0;
            do
            {
                var derivative = Derivatives[lastDerIdx - m];
                var lmscof = (float)GetLmsCoefficient(order, stepIndex, m);
                // Multiply to coeff by each derivatives to create the new tensors
                lmsDerProduct.InPlaceContinueWith((i, a) =>
                    a +
                    (derivative.GetValue(i) * lmscof)
                );
            } while (++m < iter);

            lmsDerProduct.EvaluateAndWriteTo(sample);
            var prevSample = sample;
        }
    }
}
