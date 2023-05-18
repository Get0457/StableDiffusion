﻿using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion.ML.OnnxRuntime
{
    public static class SafetyChecker
    {
        public static int IsSafe(Tensor<float> resultImage, StableDiffusionConfig config)
        {

            var sessionOptions = config.GetSessionOptionsForEp();
            using var safetySession = new InferenceSession(config.SafetyModelPath, sessionOptions);

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("clip_input", resultImage)};
            
            // Run session and send the input data in to get inference output. 
            var output = safetySession.Run(input);
            var result = ((DenseTensor<int>)output.First().Value)[0];

            return result;
        }
    }
}
