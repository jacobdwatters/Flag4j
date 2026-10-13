/*
 * MIT License
 *
 * Copyright (c) 2026. Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4jv3.linalg.internal.kernels;

import org.flag4jv3.ndarrays.Layout;
import org.flag4jv3.ndarrays.Shape;
import org.flag4jv3.ndarrays.dense.DenseData;
import org.flag4jv3.ndarrays.dense.DenseDataF64;
import org.flag4jv3.ndarrays.dense.DenseDoubleData;
import org.flag4jv3.util.NewValidateParameters;
import org.flag4jv3.util.tuples.IntPair;

import java.util.Arrays;
import java.util.Objects;

import static org.flag4jv3.linalg.internal.kernels.dense.DenseCopy.compactCopy;

// TODO NOW: Improved docs.
public final class DenseKernelSupport {

    private DenseKernelSupport() {
        // Hide default constructor for utility class.
    }


    /// Resolves the source data for an element-wise operation on a dense nD-array.
    ///
    /// Specifically, this method guarantees that the returned [dense data][DenseData] will not result in collisions
    /// when applying an element-wise operation on `src` and writing the result to `out`.
    ///
    /// @param src The source data for the element-wise operation.
    /// @param out The output data for the element-wise operation.
    /// @param <T> The type of the data being operated on.
    /// @return [Dense data][DenseData] that can be safely operated on and written to `out`. This will either be a newly constructed
    /// [DenseData] object or a reference to `src` depending on the conditions mentioned above.
    public static <T> DenseData<T> resolveSrc(DenseData<T> src, DenseData<T> out) {
        if (src.buffer() != out.buffer()) {
            return src; // The array buffers are not the same and so can't share memory.
        }

        if (src.layout().offset() == out.layout().offset()
                && Arrays.equals(src.layout().strides(), out.layout().strides())) {
            return src; // Offsets and strides are exactly equal, no risk of clobbering past results in element-wise operations.
        }

        var srcMinMax = src.layout().minMaxIndex();
        var outMinMax = out.layout().minMaxIndex();
        if (srcMinMax.second() < outMinMax.first() || outMinMax.second() < srcMinMax.first()) {
            return src; // The occupied extents of the buffer for each layout are disjoint (i.e., non-overlapping).
        }

        return compactCopy(src);
    }


    /// Resolves the source data for an element-wise operation on a dense nD-array.
    ///
    /// Specifically, this method guarantees that the returned [dense data][DenseDoubleData] will not result in collisions
    /// when applying an element-wise operation on `src` and writing the result to `out`.
    ///
    /// @param src The source data for the element-wise operation.
    /// @param out The output data for the element-wise operation.
    /// @param <T> The type of the data being operated on.
    /// @return [Dense data][DenseDoubleData] that can be safely operated on and written to `out`. This will either be a newly constructed
    /// [DenseDoubleData] object or a reference to `src` depending on the conditions mentioned above.
    public static DenseDoubleData resolveSrc(DenseDoubleData src, DenseDoubleData out) {
        if (src.buffer() != out.buffer()) {
            return src; // The array buffers are not the same and so can't share memory.
        }

        if (src.layout().offset() == out.layout().offset()
                && Arrays.equals(src.layout().strides(), out.layout().strides())) {
            return src; // Offsets and strides are exactly equal, no risk of clobbering past results in element-wise operations.
        }

        var srcMinMax = src.layout().minMaxIndex();
        var outMinMax = out.layout().minMaxIndex();
        if (srcMinMax.second() < outMinMax.first() || outMinMax.second() < srcMinMax.first()) {
            return src; // The occupied extents of the buffer for each layout are disjoint (i.e., non-overlapping).
        }

        return compactCopy(src);
    }


    public static DenseDataF64 resolveOut(Shape resultShape, DenseDataF64 out) {
        Objects.requireNonNull(resultShape, "resultShape must not be null.");

        if (out == null) {
            Layout layout = Layout.contiguous(resultShape);
            return new DenseDataF64(layout, new double[bufferLen(layout, 1)]);
        }

        requireSameShape(out.layout().shape(), resultShape);
        requireWritable(out.layout());
        return out;
    }


    static int bufferLen(Layout layout, int componentsPerSlot) {
        return Math.multiplyExact(layout.shape().numelIntValueExact(), componentsPerSlot);
    }


    /// Checks that a buffer with the specified length can store a layout.
    ///
    /// @param bufferLength The length of the buffer of interest.
    /// @param layout The layout to be stored.
    /// @throws IllegalArgumentException If a buffer with length `bufferLength` cannot store the specified `layout`
    public static void requireCapacity(int bufferLength, Layout layout, int componentsPerSlot) {
        NewValidateParameters.ensureSign(componentsPerSlot, NewValidateParameters.Sign.POSITIVE, "componentsPerSlot must be positive.");
        if (layout.shape().hasZeroAxis()) return;

        IntPair mm = layout.minMaxIndex();   // element units, inclusive
        long lo = (long) mm.first()*componentsPerSlot;
        long hi = ((long) mm.second() + 1)*componentsPerSlot;   // exclusive

        if (lo < 0 || hi > bufferLength) {
            throw new IllegalArgumentException("buffer length " + bufferLength + " cannot hold layout reaching ["
                    + lo + ", " + hi + ") in buffer units (componentsPerSlot=" + componentsPerSlot + ").");
        }
    }


    // ---- opt-in: kernel can't run in place ----------------------------------
    public static void requireNoAlias(Object src, Object dest) {
        if (src == dest) {
            throw new IllegalArgumentException(
                    "this operation cannot be performed in-place; src and dest must be distinct buffers.");
        }
    }


    // ---- opt-in: catches a stray broadcast passed as a write target ---------
    public static void requireWritable(Layout layout) {
        if (!layout.isWritable()) {
            throw new IllegalArgumentException("layout must writeable.");
        }
    }


    /// Ensures that two [layout's][Layout] shapes are [equal][Shape#equals(java.lang.Object)].
    ///
    /// @param a The first layout to compare.
    /// @param b The second layout to compare.
    /// @throws IllegalArgumentException If `a` and `b` do not have matching shapes.
    public static void requireSameShape(Layout a, Layout b) {
        requireSameShape(a.shape(), b.shape());
    }


    /// Ensures that two shapes are [equal][Shape#equals(java.lang.Object)].
    ///
    /// @param a The first shape to compare.
    /// @param b The second shape to compare.
    /// @throws IllegalArgumentException If `a` and `b` are not [equal][Shape#equals(java.lang.Object)].
    public static void requireSameShape(Shape a, Shape b) {
        if (!a.equals(b)) {
            throw new IllegalArgumentException("Expecting shapes to match but got " + a + " and " + b);
        }
    }


    public static int coalesceCOrder(int[] dims, int[] strides, int rank, int[] outDims, int[] outStrides) {
        int n = 0;
        for (int i = 0; i < rank; i++) {
            if (dims[i] == 0) {
                outDims[0] = 0;
                outStrides[0] = 1;
                return 1;
            }  // empty
            if (dims[i] == 1) continue;   // extent-1 axes: stride is arbitrary, and
            // keeping them would block otherwise-valid merges
            outDims[n] = dims[i];
            outStrides[n] = strides[i];
            n++;
        }
        if (n == 0) {
            outDims[0] = 1;
            outStrides[0] = 1;
            return 1;
        }  // scalar / all-ones

        // Merge adjacent axes, innermost outward. Axis i absorbs into the current
        // group iff stepping I once equals stepping the whole group.
        int last = n - 1;
        for (int i = n - 2; i >= 0; i--) {
            if (outStrides[i] == outStrides[last]*outDims[last]) {
                outDims[last] *= outDims[i];      // group stride stays the inner stride
            } else {
                last--;
                outDims[last] = outDims[i];
                outStrides[last] = outStrides[i];
            }
        }

        int count = n - last;
        if (last > 0) { // shift the result down to slice 0
            System.arraycopy(outDims, last, outDims, 0, count);
            System.arraycopy(outStrides, last, outStrides, 0, count);
        }

        return count;
    }


    public static int coalesceCOrder2(int[] dims, int[] stridesA, int[] stridesB, int rank,
                                      int[] outDims, int[] outSA, int[] outSB) {
        int n = 0;
        for (int i = 0; i < rank; i++) {
            if (dims[i] == 0) {
                outDims[0] = 0;
                outSA[0] = 1;
                outSB[0] = 1;
                return 1;
            }
            if (dims[i] == 1) continue;
            outDims[n] = dims[i];
            outSA[n] = stridesA[i];
            outSB[n] = stridesB[i];
            n++;
        }
        if (n == 0) {  // scalar / all-ones
            outDims[0] = 1;
            outSA[0] = 1;
            outSB[0] = 1;
            return 1;
        }

        int last = n - 1;
        for (int i = n - 2; i >= 0; i--) {
            long span = (long) outDims[last];
            if (outSA[i] == outSA[last]*span && outSB[i] == outSB[last]*span) {
                outDims[last] *= outDims[i];
            } else {
                last--;
                outDims[last] = outDims[i];
                outSA[last] = outSA[i];
                outSB[last] = outSB[i];
            }
        }

        int count = n - last;
        if (last > 0) {
            System.arraycopy(outDims, last, outDims, 0, count);
            System.arraycopy(outSA, last, outSA, 0, count);
            System.arraycopy(outSB, last, outSB, 0, count);
        }
        return count;
    }


    public static void insertionSortByStride() {

    }
}
