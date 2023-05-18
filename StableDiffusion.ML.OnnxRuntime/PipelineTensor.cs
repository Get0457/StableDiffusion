using Microsoft.ML.OnnxRuntime.Tensors;
using StableDiffusion.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace StableDiffusion
{
    internal class PipelineTensor<T> : PipelineTensor<T, T>
    {
        public PipelineTensor(Tensor<T> tensor, Func<T, T> func) : base(tensor, func)
        {
        }
        public PipelineTensor(Tensor<T> tensor, Func<int, T, T> func) : base(tensor, func)
        {
        }
        public PipelineTensor(Tensor<T> tensor) : this(tensor, x => x)
        {
        }
        public PipelineTensor<T> ContinueWith(Func<T, T> func)
        {
            return new PipelineTensor<T>(this, func);
        }
        public PipelineTensor<T> ContinueWith(Func<int, T, T> func)
        {
            return new PipelineTensor<T>(this, func);
        }
        public PipelineTensor<T> InPlaceContinueWith(Func<int, T, T> func)
        {
            var oldFunc = ApplyFunction;
            ApplyFunction = (i, x) => func(i, oldFunc(i, x));
            return this;
        }
        public PipelineTensor<T> InPlaceContinueWith(Func<T, T> func)
        {
            var oldFunc = ApplyFunction;
            ApplyFunction = (i, x) => func(oldFunc(i, x));
            return this;
        }
    }
    internal class PipelineTensor<TIn, TOut> : Tensor<TOut>
    {
        protected Func<int, TIn, TOut> ApplyFunction;
        protected Tensor<TIn> OriginalTensor;
        public PipelineTensor(Tensor<TIn> tensor, Func<TIn, TOut> func) : this(tensor, (_, x) => func(x))
        {
            
        }
        public PipelineTensor(Tensor<TIn> tensor, Func<int, TIn, TOut> func) : base(tensor.Dimensions, false)
        {
            OriginalTensor = tensor;
            ApplyFunction = func;
        }

        public DenseTensor<TOut> Evaluate()
        {
            var denseTensor = new DenseTensor<TOut>(Dimensions);
            EvaluateAndWriteTo(denseTensor);
            return denseTensor;
        }
        public void EvaluateAndWriteTo(Tensor<TOut> tensor)
        {
            for (int i = 0; i < Length; i++)
            {
                tensor.SetValue(i, GetValue(i));
            }
        }
        public override Tensor<TOut> Clone() => Evaluate();

        public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        {
            return new DenseTensor<TResult>(Dimensions);
        }

        public override TOut GetValue(int index)
        {
            return ApplyFunction(index, OriginalTensor.GetValue(index));
        }

        public override Tensor<TOut> Reshape(ReadOnlySpan<int> dimensions)
        {
            return new ReshapedTensor<TOut>(this, dimensions);
        }

        public override void SetValue(int index, TOut value)
        {
            throw new InvalidOperationException("Write is not supported on Pipeline Tensor, please evaluate the tensor as a copy");
        }
        public PipelineTensor<TOut, TOut2> ContinueWith<TOut2>(Func<TOut, TOut2> func)
        {
            return new PipelineTensor<TOut, TOut2>(this, func);
        }
    }
}
