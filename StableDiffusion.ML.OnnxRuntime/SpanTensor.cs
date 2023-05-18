using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace StableDiffusion.ML.OnnxRuntime
{
    internal class SpanTensor<T> : Tensor<T>
    {
        static ReadOnlySpan<int> ToDimensionLength(ReadOnlySpan<Range> r, Tensor<T> tensor, out (int Offset, int Length)[] OffsetLength)
        {
            int[] dim = new int[r.Length];
            OffsetLength = new (int, int)[r.Length];
            for (int i = 0; i < r.Length; i++)
            {
                OffsetLength[i] = r[i].GetOffsetAndLength(tensor.Dimensions[i]);
                dim[i] = OffsetLength[i].Length;
            }
            return dim;
        }
        Tensor<T> InternalTensor;
        (int Offset, int Length)[] OffsetLength;
        public SpanTensor(Tensor<T> Tensor, ReadOnlySpan<Range> dimensions) : base(ToDimensionLength(dimensions, Tensor, out var OffsetLength), false)
        {
            this.OffsetLength = OffsetLength;
            InternalTensor = Tensor;
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
            return InternalTensor[GetIndices(index)];
        }

        public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions)
        {
            return new ReshapedTensor<T>(this, dimensions);
        }

        public override void SetValue(int index, T value)
        {
            InternalTensor[GetIndices(index)] = value;
        }



        /// <summary>
        /// Calculates the n-d indices from the 1-d index in a layout specificed by strides
        /// </summary>
        /// <param name="strides"></param>
        /// <param name="reverseStride"></param>
        /// <param name="index"></param>
        /// <param name="indices"></param>
        /// <param name="startFromDimension"></param>
        int[] GetIndices(int index)
        {
            int[] indices = new int[Strides.Length];

            Debug.Assert(Strides.Length == indices.Length);

            // scalar tensor - nothing to process
            if (indices.Length == 0)
            {
                return indices;
            }

            int remainder = index;
            for (int i = 0; i < Strides.Length; i++)
            {
                var stride = Strides[i];
                indices[i] = remainder / stride + OffsetLength[i].Offset;
                remainder %= stride;
            }
            return indices;
        }
    }


}
