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

package org.flag4j.core_temp.structures.rings;

/**
 * <p>This interface specifies a mathematical semi-ring.</p>
 *
 * <p>SemiRing elements should be immutable.</p>
 *
 * <p>Formally, an semi-ring is a set <b>R</b> with the binary operations_old addition (+) and multiplication (*)
 * defined such that for elements a, b, c in <b>R</b> the following are satisfied:
 *  <ul>
 *      <li>Addition and multiplication are associative: a + (b + c) = (a + b) + c and a * (b * c) = (a * b) * c.</li>
 *      <li>Addition is commutative: a + b = b + a</li>
 *      <li>Existince of additive and multiplicitive identities: There exisits two distinct elements 0 and 1 in <b>R</b> sucht that a + 0 = 0
 *      and a * 1 = 1 (called the addative and multiplicitive identities respectively).</li>
 *      <li>Distributivity of multiplication over addition: a * (b + c) = (a * b) + (a * c).</li>
 *  </ul>
 * </p>
 *
 * <p>Semi-rings generalize {@link Ring rings} in that additive inverses need not exists.</p>
 *
 * @param <T> Type of the field element.
 * @see org.flag4j.core_temp.structures.fields.Field
 * @see org.flag4j.core_temp.structures.fields.GeneralizedRealField
 * @see Ring
 */
public interface SemiRing<T extends SemiRing<T>> {
    /**
     * Sums two elements of this semi-ring (associative and commutative).
     * @param b Second semi-ring element in sum.
     * @return The sum of this element and {@code b}.
     */
    public T add(T b);


    /**
     * Multiplies two elements of this semi-ring (associative).
     * @param b Second semi-ring element in product.
     * @return The product of this semi-ring element and {@code b}.
     */
    public T mult(T b);


    /**
     * Compares this element of the semi-ring with {@code b}.
     * @param b Second element of the semi-ring.
     * @return An int value:
     * <ul>
     *     <li>0 if this semi-ring element is equal to {@code b}.</li>
     *     <li>< 0 if this semi-ring element is less than {@code b}.</li>
     *     <li>> 0 if this semi-ring element is greater than {@code b}.</li>
     *     Hence, this method returns zero if and only if the two semi-ring elemetns are equal, a negative value if and only the semi-ring
     *     element it was called on is less than {@code b} and positive if and only if the semi-ring element it was called on is greater
     *     than {@code b}.
     * </ul>
     */
    public int compareTo(T b);
}
