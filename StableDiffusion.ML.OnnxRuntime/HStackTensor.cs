using Microsoft.ML.OnnxRuntime.Tensors;
using StableDiffusion.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace StableDiffusion
{
    internal class HStackTensor<T> : Tensor<T>
    {
        Memory<Tensor<T>> Tensors;
        public HStackTensor(params Tensor<T>[] tensors) : base(
            (ReadOnlySpan<int>)Enumerable.Repeat(tensors.Length, 1).Concat(tensors[0].Dimensions.ToArray().Skip(1)).ToArray()
        , false)
        {
            Tensors = tensors;
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
            long idx = index;
            var enumerator = Tensors.Span.GetEnumerator();
            while (enumerator.MoveNext())
            {
                if (idx >= enumerator.Current.Length)
                {
                    idx -= enumerator.Current.Length;
                    continue;
                }
                return enumerator.Current.GetValue((int)idx);
            }
            throw new ArgumentOutOfRangeException();
        }

        public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions)
        {
            return new ReshapedTensor<T>(this, dimensions);
        }

        public override void SetValue(int index, T value)
        {
            long idx = index;
            var enumerator = Tensors.Span.GetEnumerator();
            while (enumerator.MoveNext())
            {
                if (idx >= enumerator.Current.Length)
                {
                    idx -= enumerator.Current.Length;
                    continue;
                }
                enumerator.Current.SetValue((int)idx, value);
                return;
            }
            throw new ArgumentOutOfRangeException();
        }
    }
}
