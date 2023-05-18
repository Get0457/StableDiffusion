using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class TensorHelper
    {
        public static DenseTensor<float> DivideTensorByFloat(float[] data, float value, int[] dimensions)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = data[i] / value;
            }

            return new(data, dimensions);
        }

        public static DenseTensor<float> MultipleTensorByFloat(float[] data, float value, int[] dimensions)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = data[i] * value;
            }

            return new(data, dimensions);
        }

        public static DenseTensor<float> MultipleTensorByFloat(Tensor<float> data, float value)
        {
            return MultipleTensorByFloat(data.ToArray(), value, data.Dimensions.ToArray());
        }

        public static DenseTensor<float> AddTensors(float[] sample, float[] sumTensor, int[] dimensions)
        {
            for(var i=0; i < sample.Length; i++)
            {
                sample[i] = sample[i] + sumTensor[i];
            }
            return new(sample, dimensions); ;
        }

        public static DenseTensor<float> AddTensors(Tensor<float> sample, Tensor<float> sumTensor)
        {
            return AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());
        }

        public static (Tensor<float>, Tensor<float>) SplitTensor(Tensor<float> tensorToSplit)
        {
            return (new SpanTensor<float>(tensorToSplit, DimRangeOf(..1, .., .., ..)),
                new SpanTensor<float>(tensorToSplit, DimRangeOf(1.., .., .., ..)));

        }

        public static DenseTensor<float> SumTensors(Tensor<float>[] tensorArray, int[] dimensions)
        {
            var sumTensor = new DenseTensor<float>(dimensions);
            var sumArray = new float[sumTensor.Length];

            for (int m = 0; m < tensorArray.Count(); m++)
            {
                var tensorToSum = tensorArray[m].ToArray();
                for (var i = 0; i < tensorToSum.Length; i++)
                {
                    sumArray[i] += (float)tensorToSum[i];
                }
            }

            return new(sumArray, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(float[] sample, float[] subTensor, int[] dimensions)
        {
            for (var i = 0; i < sample.Length; i++)
            {
                sample[i] = sample[i] - subTensor[i];
            }
            return new(sample, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(Tensor<float> sample, Tensor<float> subTensor)
        {
            return SubtractTensors(sample.ToArray(), subTensor.ToArray(), sample.Dimensions.ToArray());
        }

        public static Tensor<float> GetRandomTensor(ReadOnlySpan<int> dimensions)
        {
            var random = new Random();
            var latents = new DenseTensor<float>(dimensions);
            var latentsArray = latents.ToArray();

            for (int i = 0; i < latentsArray.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number
                latentsArray[i] = (float)standardNormalRand;
            }

            latents = new(latentsArray, latents.Dimensions.ToArray());

            return latents;

        }
        static ReadOnlySpan<Range> DimRangeOf(params Range[] ranges) => ranges;
    }
}
