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

package org.flag4j.core_temp;


import org.flag4j.core.Shape;

/**
 * <p>This abstract class defines a tensor whose elements satisfy the axioms of a field.</p>
 *
 * <p>To allow for primitive types, the elements of this tensor do not neccesarily have to implement
 * {@link org.flag4j.core_temp.structures.rings.SemiRing}</p>.
 *
 * <p>Formally, an ring is a set <b>R</b> with the binary operations_old addition (+) and multiplication (*)
 * defined such that for elements a, b, c in <b>R</b> the following are satisfied:
 *  <ul>
 *      <li>Addition and multiplication are associative: a + (b + c) = (a + b) + c and a * (b * c) = (a * b) * c.</li>
 *      <li>Addition is commutative: a + b = b + a</li>
 *      <li>Existince of additive and multiplicitive identities: There exisits two distinct elements 0 and 1 in <b>R</b> sucht that a + 0 = 0
 *      and a * 1 = 1 (called the addative and multiplicitive identities respectively).</li>
 *      <li>Existince of addative inverse: There exists an element -a in <b>R</b> such that a + (-a) = 0.</li>
 *      <li>Distributivity of multiplication over addition: a * (b + c) = (a * b) + (a * c).</li>
 *  </ul>
 * </p>
 *
 * @param <T> Type of this tensor.
 * @param <U> Storage for entries of this tensor.
 * @param <V> Type (or wrapper) of an element of this tensor. Should satisfy the axioms of a ring as stated.
 */
public abstract class TensorOverField<T extends TensorOverField<T, U, V>, U, V> extends TensorOverRing<T, U, V> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected TensorOverField(Shape shape, U entries) {
        super(shape, entries);
    }


    /**
     * Computes the element-wise square root of a tensor.
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    public abstract T sqrt();


    /**
     * Computes the element-wise reciprocals of this tensor.
     * @return A tensor containing the reciprocal elements of this tensor.
     */
    public abstract T recip();


    /**
     * Divides each entry of this tensor by s scalar field element.
     *
     * @param b Scalar field value in quotient.
     *
     * @return The quotient of this tensor with {@code b}.
     */
    public abstract T div(V b);
}
