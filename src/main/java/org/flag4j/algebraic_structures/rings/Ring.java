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

package org.flag4j.algebraic_structures.rings;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.semirings.Semiring;

/**
 * <p>This interface specifies a mathematical ring. This interface not only meets the basic definition of a ring,
 * but also specifies some additional operations which are common and useful.
 *
 * <p>Ring elements should be immutable.
 *
 * <p>Formally, a ring is a set <b>R</b> with the binary operations addition (+) and multiplication (*)
 * defined such that for elements a, b, c in <b>R</b> the following are satisfied:
 *  <ul>
 *      <li>Addition and multiplication are associative: a + (b + c) = (a + b) + c and a * (b * c) = (a * b) * c.</li>
 *      <li>Addition is commutative: a + b = b + a</li>
 *      <li>Existence of additive and multiplicative identities: There exists two distinct elements 0 and 1 in <b>R</b> such that a + 0 = 0
 *      and a * 1 = 1 (called the additive and multiplicative identities respectively).</li>
 *      <li>Existence of additive inverse: There exists an element -a in <b>R</b> such that a + (-a) = 0.</li>
 *      <li>Distributivity of multiplication over addition: a * (b + c) = (a * b) + (a * c).</li>
 *  </ul>
 * 
 *
 * <p>Rings generalize {@link Field fields} in that multiplication need not be
 * commutative and multiplicative inverses need not exists. As a result, division is not defined for a ring.
 *
 * @param <T> Type of the ring element.
 * @see Field
 * @see Semiring
 */
public interface Ring<T extends Ring<T>> extends Semiring<T> {

    /**
     * Computes difference of two elements of this ring.
     * @param b Second ring element in difference.
     * @return The difference of this ring element and {@code b}.
     */
    public T sub(T b);


    /**
     * <p>Computes the additive inverse for an element of this ring.
     *
     * <p>An element -x is an additive inverse for a filed element x if -x + x = 0 where 0 is the additive identity.
     *
     * @return The additive inverse for this ring element.
     */
    public T addInv();


    /**
     * <p>Computes the absolute value of this ring element.
     *
     * @return The absolute value of this ring element.
     * @implNote By default, this is implemented as {@code return }{@link #mag()}{@code ;}
     */
    default double abs() {
        return mag();
    }


    /**
     * Computes the magnitude of this ring element.
     * @return The magnitude of this ring element.
     */
    public double mag();


    /**
     * Computes the conjugation of this ring element.
     * @return The conjugation of this ring element.
     * @implNote The default implementation of this method simply returns this ring element.
     */
    public default T conj() {
        return (T) this;
    }
}
