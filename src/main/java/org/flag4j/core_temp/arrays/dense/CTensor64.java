/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.core_temp.arrays.dense;


// TODO: Needs to implement DenseTensorMixin (Should probably have a DenseFieldTensorBase abstract class as well).
/**
 * Complex dense tensor backed by an array of {@link org.flag4j.core_temp.structures.fields.Complex64}'s.
 */
//public class CTensor64 extends DenseFieldTensorBase<CTensor64, CooCTensor64, Complex64> {
//
//    /**
//     * Creates a tensor with the specified entries and shape.
//     *
//     * @param shape Shape of this tensor.
//     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
//     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
//     */
//    public CTensor64(Shape shape, Complex64[] entries) {
//        super(shape, entries);
//    }
//
//
//    /**
//     * Creates a zero tensor with the specified shape.
//     *
//     * @param shape Shape of this tensor.
//     */
//    public CTensor64(Shape shape) {
//        super(shape, new Complex64[shape.totalEntries().intValueExact()]);
//        Arrays.fill(entries, Complex64.ZERO);
//    }
//
//
//    /**
//     * Creates a tensor with the specified shape and filled with {@code fillValue}.
//     *
//     * @param shape Shape of this tensor.
//     * @param fillValue Value to fill this tensor with.
//     */
//    public CTensor64(Shape shape, Complex64 fillValue) {
//        super(shape, new Complex64[shape.totalEntries().intValueExact()]);
//        Arrays.fill(entries, fillValue);
//    }
//
//
//    /**
//     * Creates a tensor with the specified shape and filled with {@code fillValue}.
//     *
//     * @param shape Shape of this tensor.
//     * @param fillValue Value to fill this tensor with. Must be a string representation of a complex number.
//     */
//    public CTensor64(Shape shape, String fillValue) {
//        super(shape, new Complex64[shape.totalEntries().intValueExact()]);
//        Arrays.fill(entries, new Complex64(fillValue));
//    }
//
//
//    /**
//     * <p>Creates a tensor with the specified entries and shape.</p>
//     * <p>String array must contain properly formatted string representation of complex numbers.</p>
//     *
//     * @param shape Shape of this tensor.
//     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
//     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
//     * @throws ComplexNumberParsingException If any entry in {@code entries} is not a properly formatted
//     * string representation of complex number.
//     */
//    public CTensor64(Shape shape, String[] entries) {
//        super(shape, new Complex64[entries.length]);
//
//        // Parse string values.
//        for(int i=0, size=entries.length; i<size; i++)
//            super.entries[i] = new Complex64(entries[i]);
//    }
//}
