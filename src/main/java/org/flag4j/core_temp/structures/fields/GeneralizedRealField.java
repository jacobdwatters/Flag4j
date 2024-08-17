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

package org.flag4j.core_temp.structures.fields;


/**
 * This interface specifies a {@link Field} which is a superset of the real numbers. It specifies operations_old between the field element
 * and a primative double.
 * 
 * @param <T> Type of the field element
 */
public interface GeneralizedRealField<T extends GeneralizedRealField<T>> extends Field<T> {

    /**
     * Sums an element of this field with a real number (associative and commutative).
     * @param b Real number in sum.
     * @return The sum of this element and {@code b}.
     */
    public T add(double b);


    /**
     * Computes difference of an element of this field and a real number.
     * @param b Real number in difference.
     * @return The difference of this field element and {@code b}.
     */
    public T sub(double b);


    /**
     * Multiplies an element of this field and a real number (associative and commutative).
     * @param b Real number product.
     * @return The product of this field element and {@code b}.
     */
    public T mult(double b);


    /**
     * Computes the quotient of a element of this field and a real number.
     * @param b Real number quotient.
     * @return The quotient of this field element and {@code b}.
     */
    public T div(double b);
}
