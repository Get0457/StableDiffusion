using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace StableDiffusion.ML.OnnxRuntime
{
    class ReshapedTensor<T> : Tensor<T>
    {
        Tensor<T> originalTensor;
        public ReshapedTensor(Tensor<T> originalTensor, ReadOnlySpan<int> dimensions) : base(dimensions, false)
        {
            this.originalTensor = originalTensor;
        }

        public override Tensor<T> Clone()
        {
            var denseTensor = CloneEmpty<T>();
            for (int i = 0; i < Length; i++)
            {
                denseTensor.SetValue(i, GetValue(i));
            }
            return denseTensor;
        }

        public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        {
            return new DenseTensor<TResult>(Dimensions);
        }

        public override T GetValue(int index)
        {
            return originalTensor.GetValue(index);
        }

        public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions)
        {
            // using originalTensor to reduce recursion amount
            return new ReshapedTensor<T>(originalTensor, dimensions);
        }

        public override void SetValue(int index, T value)
        {
            originalTensor.SetValue(index, value);
        }
    }
}
