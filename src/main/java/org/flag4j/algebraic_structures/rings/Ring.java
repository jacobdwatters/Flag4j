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
import org.flag4j.algebraic_structures.semi_rings.SemiRing;

/**
 * <p>This interface specifies a mathematical ring. This interface not only meets the basic definition of a ring,
 * but also specifies some additional operations_old which are common and useful.</p>
 *
 * <p>Ring elements should be immutable.</p>
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
 * <p>Rings generalize {@link Field fields} in that multiplication need not be
 * commutative and multiplicitive inverses need not exists. As a result, division is not defined for a ring.</p>
 *
 * @param <T> Type of the ring element.
 * @see Field
 * @see SemiRing
 */
public interface Ring<T extends Ring<T>> extends SemiRing<T> {

    /**
     * Computes difference of two elements of this ring.
     * @param b Second ring element in difference.
     * @return The difference of this ring element and {@code b}.
     */
    public T sub(T b);


    /**
     * <p>Computes the addative inverse for an element of this ring.</p>
     *
     * <p>An element -x is an addative inverse for a filed element x if -x + x = 0 where 0 is the addative identity..</p>
     *
     * @return The additive inverse for this ring element.
     */
    public T addInv();


    /**
     * <p>Computes the absolute value of this ring element.</p>
     *
     * @return The absolute value of this ring element.
     * @implNote By default, this is implemented as {@code return }{@link #mag()}{@code ;}
     */
    default double abs() {
        return mag();
    }


    /**
     * Computes the magnitude of this ring element.
     * @return The magniitude of this ring element.
     */
    public double mag();


    /**
     * Computs the conjugation of this ring element.
     * @return The conjugation of this ring element.
     * @implNote The default implementation of this method simply returns this ring element.
     */
    public default T conj() {
        return (T) this;
    }
}
