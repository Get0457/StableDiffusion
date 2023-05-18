using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace StableDiffusion.ML.OnnxRuntime
{
    public static class TextProcessing
    {
        public static Tensor<float> PreprocessText(String prompt, StableDiffusionConfig config)
        {
            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TokenizeText(prompt, config);
            var textPromptEmbeddings = TextEncoder(textTokenized, config);

            // Create uncond_input of blank tokens
            var uncondInputTokens = CreateUncondInput();
            var uncondEmbedding = TextEncoder(uncondInputTokens, config);

            // Concant textEmeddings and uncondEmbedding
            //DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });

            Tensor<float> textEmbeddings = new HStackTensor<float>(uncondEmbedding, textPromptEmbeddings);

            //for (var i = 0; i < textPromptEmbeddings.Length; i++)
            //{
            //    textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
            //    textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
            //}

            GC.Collect();
            return textEmbeddings;
        }


        const int modelMaxLength = 77;
        const int blankTokenValue = 49407;
        public static Tensor<int> TokenizeText(string text, StableDiffusionConfig config)
        {
            // Create session options for custom op of extensions
            var sessionOptions = new SessionOptions();
            sessionOptions.RegisterCustomOpLibraryV2(config.OrtExtensionsPath, out var libraryHandle);
            
            // Create an InferenceSession from the onnx clip tokenizer.
            using var tokenizeSession = new InferenceSession(config.TokenizerOnnxPath, sessionOptions);
            // Run session and send the input data in to get inference output. 
            var tokens = tokenizeSession.Run(new NamedOnnxValue[] {
                NamedOnnxValue.CreateFromTensor(
                    "string_input",
                    new DenseTensor<string>(new string[] { text }, DimensionOf(1)
                )) 
            });


            var inputIds = (DenseTensor<long>)tokens.First().Value;

            // Cast inputIds to Int32
            var InputIdsInt = new DenseTensor<int>(new int[modelMaxLength], DimensionOf(77));
            int i = 0;
            foreach (var item in inputIds)
            {
                if (i >= modelMaxLength) break;
                InputIdsInt.SetValue(i++, (int)item);
            }
            // Pad array with 49407 until length is modelMaxLength
            while (i < modelMaxLength)
            {
                InputIdsInt.SetValue(i++, blankTokenValue);
            }
            
            return InputIdsInt;
        }

        public static DenseTensor<int> CreateUncondInput()
        {
            var InputIdsInt = new DenseTensor<int>(new int[modelMaxLength], DimensionOf(77));
            int i = 0;
            InputIdsInt.SetValue(i++, 49406);
            // Pad array with 49407 until length is modelMaxLength
            while (i < modelMaxLength)
            {
                InputIdsInt.SetValue(i++, blankTokenValue);
            }

            return InputIdsInt;
        }
        public static Tensor<float> TextEncoder(Tensor<int> tokenizedInput, StableDiffusionConfig config)
        {
            // Create input tensor.
            var input_ids = tokenizedInput.Reshape(DimensionOf(1, tokenizedInput.Dimensions[0]));

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids) };

            // Set CUDA EP
            var sessionOptions = config.GetSessionOptionsForEp();

            using var encodeSession = new InferenceSession(config.TextEncoderOnnxPath, sessionOptions);
            // Run inference.
            using var encoded = encodeSession.Run(input);
            return ((DenseTensor<float>)encoded.First().Value).Clone();

        }
        static int[] DimensionOf(params int[] Is) => Is;
    }
}