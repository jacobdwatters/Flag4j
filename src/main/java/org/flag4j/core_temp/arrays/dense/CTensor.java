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

import org.flag4j.core.Shape;
import org.flag4j.core_temp.TensorPrimitiveOpsMixin;
import org.flag4j.core_temp.structures.fields.Complex128;


/**
 * Complex dense tensor backed by an array of {@link Complex128}'s.
 */
public class CTensor extends ComplexFieldTensor<Complex128> implements TensorPrimitiveOpsMixin<FieldTensor<Complex128>> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public CTensor(Shape shape, Complex128[] entries) {
        super(shape, entries);
    }


    /**
     * <p>Creates a tensor with the specified entries and shape.</p>
     * <p>String array must contain properly formatted string representation of complex numbers.</p>
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     * @throws org.flag4j.util.exceptions.ComplexNumberParseingException If any entry in {@code entries} is not a properly formatted
     * string representation of complex number.
     */
    public CTensor(Shape shape, String[] entries) {
        super(shape, new Complex128[entries.length]);

        // Parse string values as .
        for(int i=0, size=entries.length; i<size; i++) {
            super.entries[i] = new Complex128(entries[i]);
        }
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public CTensor(Shape shape, double[] entries) {
        super(shape, new Complex128[entries.length]);

        // Wrap values as complex values.
        for(int i=0, size=entries.length; i<size; i++) {
            super.entries[i] = new Complex128(entries[i]);
        }
    }


    /**
     * Adds a scalar value to each element of this tensor.
     *
     * @param b Value to add to each entry of this tensor.
     *
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    @Override
    public CTensor add(double b) {
        Complex128[] sum = new Complex128[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            sum[i] = entries[i].add(b);
        }

        return new CTensor(shape, sum);
    }


    /**
     * Subtracts a scalar value from each element of this tensor.
     *
     * @param b Value to subtract from each entry of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    @Override
    public CTensor sub(double b) {
        Complex128[] diff = new Complex128[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            diff[i] = entries[i].sub(b);
        }

        return new CTensor(shape, diff);
    }


    /**
     * Computes the sclar multiplication between this tensor and the specified scalar {@code factor}.
     *
     * @param factor Scalar factor to apply to this tensor.
     *
     * @return The sclar product of this tensor and {@code factor}.
     */
    @Override
    public CTensor mult(double factor) {
        Complex128[] product = new Complex128[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            product[i] = entries[i].mult(factor);
        }

        return new CTensor(shape, product);
    }


    /**
     * Computes the scalar division of this tensor and the specified scalar {@code factor}.
     *
     * @param divisor The scalar value to divide this tensor by.
     *
     * @return The result of dividing this tensor by the specified scalar.
     */
    @Override
    public CTensor div(double divisor) {
        Complex128[] quotient = new Complex128[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            quotient[i] = entries[i].div(divisor);
        }

        return new CTensor(shape, quotient);
    }
}
