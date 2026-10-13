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

package org.flag4jv3.ndarrays.dense;

import org.flag4jv3.linalg.internal.kernels.DenseKernelSupport;
import org.flag4jv3.ndarrays.DType;
import org.flag4jv3.ndarrays.Layout;

import java.util.Objects;

public record DenseDataF64(Layout layout, double[] buffer) implements NewDenseData {

    /// DenseDataF64 consists of primitive `double`s.
    static final DType DTYPE = DType.FLOAT64;


    /// Constructs a data container for a dense nD-array of primitive `double`s.
    ///
    /// @param layout The layout of the nD-array.
    /// @param buffer The buffer containing the data.
    public DenseDataF64 {
        Objects.requireNonNull(layout);
        Objects.requireNonNull(buffer);
        DenseKernelSupport.requireCapacity(buffer.length, layout, DTYPE.componentsPerSlot());
    }


    @Override
    public Layout layout() {
        return layout;
    }


    @Override
    public DType dtype() {
        return DTYPE;
    }


    @Override
    public DenseDataF64 withLayout(Layout layout) {
        return new DenseDataF64(layout, buffer);
    }


    @Override
    public int toBufferIndex(int componentIdx, int... idxND) {
        int itemSize = DTYPE.componentsPerSlot();

        Objects.checkIndex(componentIdx, itemSize);
        return itemSize*layout.toLinearElementIndex(idxND) + componentIdx;
    }


    @Override
    public int bufferSize() {
        return buffer.length;
    }
}
