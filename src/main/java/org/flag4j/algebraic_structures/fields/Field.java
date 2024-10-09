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

package org.flag4j.algebraic_structures.fields;

import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.algebraic_structures.semi_rings.SemiRing;

/**
 * <p>This interface specifies a mathematical field. This interface not only meets the basic definition of a field,
 * but also specifies some additional operations which may not technically be part of the formal definition of a field but are common
 * and useful.</p>
 *
 * <p>Field elements should be immutable.</p>
 *
 * <p>Formally, an field is a set <b>F</b> with the binary operations addition (+) and multiplication (*)
 * defined such that for elements a, b, c in <b>F</b> the following are satisfied:
 *  <ul>
 *      <li>Addition and multiplication are associative: a + (b + c) = (a + b) + c and a * (b * c) = (a * b) * c.</li>
 *      <li>Addition and multiplication are commutative: a + b = b + a and a * b = b * a</li>
 *      <li>Existence of additive and multiplicative identities: There exists two distinct elements 0 and 1 in <b>F</b> such that a + 0 = 0
 *      and a * 1 = 1 (called the additive and multiplicative identities respectively).</li>
 *      <li>Existence of additive inverse: There exists an element -a in <b>F</b> such that a + (-a) = 0.</li>
 *      <li>Existence of multiplicative inverse: There exists an element a<sup>-1</sup> in <b>F</b> such that a * a<sup>-1</sup> = 1.</li>
 *      <li>Distributivity of multiplication over addition: a * (b + c) = (a * b) + (a * c).</li>
 *  </ul>
 * </p>
 *
 * <p>Fields are a type of {@link Ring Ring} where multiplication must be
 * commutative and multiplicative inverses must exists.</p>
 *
 * @param <T> Type of the field element.
 * @see Ring
 * @see SemiRing
 */
public interface Field<T extends Field<T>> extends Ring<T> {

    /**
     * Multiplies two elements of this field (associative and commutative).
     * @param b Second field element in product.
     * @return The product of this field element and {@code b}.
     */
    public T mult(T b);


    /**
     * Computes the quotient of two elements of this field.
     * @param b Second field element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    public T div(T b);


    /**
     * Sums an element of this field with a real number (associative and commutative).
     * @param b Real element in sum.
     * @return The sum of this element and {@code b}.
     */
    public T add(double b);


    /**
     * Computes difference of an element of this field and a real number.
     * @param b Real value in difference.
     * @return The difference of this ring element and {@code b}.
     */
    public T sub(double b);


    /**
     * Multiplies an element of this field with a real number (associative and commutative).
     * @param b Real number in product.
     * @return The product of this field element and {@code b}.
     */
    public T mult(double b);


    /**
     * Computes the quotient of an element of this field and a real number.
     * @param b Real number in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    public T div(double b);


    /**
     * <p>Computes the multiplicative inverse for an element of this field.</p>
     *
     * <p>An element x<sup>-1</sup> is a multiplicative inverse for a filed element x if x<sup>-1</sup>*x = 1 where 1 is the
     * multiplicative identity.</p>
     *
     * @return The multiplicative inverse for this field element.
     */
    public T multInv();


    // TODO: It likely does not make sense to define sqrt for a field. Some finite fields (e.g. integers mod a prime) will not
    //  have a square root for every element. This is even true for real numbers.
    /**
     * Computes the square root of this field element.
     * @return The square root of this field element.
     */
    public T sqrt();


    /**
     * Compares this element of the field with {@code b}.
     * @param b Second element of the field.
     * @return An int value:
     * <ul>
     *     <li>0 if this field element is equal to {@code b}.</li>
     *     <li>< 0 if this field element is less than {@code b}.</li>
     *     <li>> 0 if this field element is greater than {@code b}.</li>
     *     Hence, this method returns zero if and only if the two field elements are equal, a negative value if and only the field
     *     element it was called on is less than {@code b} and positive if and only if the field element it was called on is greater
     *     than {@code b}.
     * </ul>
     */
    public int compareTo(T b);


    /**
     * Checks if this field element is finite in magnitude.
     * @return True if this field element is finite in magnitude. False otherwise (i.e. infinite, NaN etc.).
     */
    public boolean isFinite();


    /**
     * Checks if this field element is infinite in magnitude.
     * @return True if this field element is infinite in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    public boolean isInfinite();


    /**
     * Checks if this field element is NaN in magnitude.
     * @return True if this field element is NaN in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    public boolean isNaN();
}
