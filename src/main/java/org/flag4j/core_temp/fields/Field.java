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

package org.flag4j.core_temp.fields;

/**
 * <p>This interface specifies an ordered mathematical field. This interface not only meets the basic definition of an ordered field,
 * but also specifies additional operations which are common and useful are specified.</p>
 *
 * <p>field elements should be immutable.</p>
 *
 * <p>Formally, an ordered field is a set <b>F</b> with the binary operations addition (+), multiplication (*), and ordering (&le)
 * defined such that for elements a, b, c in <b>F</b> the following are satisfied:
 *  <ul>
 *      <li>Addition and multiplication are associative: a + (b + c) = (a + b) + c and a * (b * c) = (a * b) * c.</li>
 *      <li>Addition and multiplication are commutative: a + b = b + a and a * b = b * a</li>
 *      <li>Existince of additive and multiplicitive identities: There exisits two distinct elements 0 and 1 in <b>F</b> sucht that a + 0 = 0
 *      and a * 1 = 1 (called the addative and multiplicitive identities respectively).</li>
 *      <li>Existince of addative inverse: There exists an element -a in <b>F</b> such that a + (-a) = 0.</li>
 *      <li>Existince of multiplicitive inverse: There exists an element a<sup>-1</sup> in <b>F</b> such that a * a<sup>-1</sup> = 1.</li>
 *      <li>Distributivity of multiplication over addition: a * (b + c) = (a * b) + (a * c).</li>
 *      <li>A total ordering exists on <b>F</b>: A total order &le exists on <b>F</b> such that
 *          <ul>
 *              <li>if a &le b then a + c &le b + c</li>
 *              <li>if 0 &le a and 0 &le b then 0 &le a * b</li>
 *          </ul>
 *      </li>
 *  </ul>
 * </p>
 *
 * @param <T> Type of the field element.
 */
public interface Field<T extends Field<T>> {

    /**
     * Sums two elements of this field (associative and commutative).
     * @param b Second field element in sum.
     * @return The sum of this element and {@code b}.
     */
    public T add(T b);


    /**
     * Computes difference of two elements of this field.
     * @param b Second field element in difference.
     * @return The difference of this field element and {@code b}.
     */
    public T sub(T b);


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
     * <p>Computes the multiplicitive inverse for an element of this field.</p>
     *
     * <p>An element x<sup>-1</sup> is a multaplicitive inverse for a filed element x if x<sup>-1</sup>*x = 1 where 1 is the
     * multiplicitive identity.</p>
     *
     * @return The multiplicitive inverse for this field element.
     */
    public T multInv();


    /**
     * <p>Computes the addative inverse for an element of this field.</p>
     *
     * <p>An element -x is an addative inverse for a filed element x if -x + x = 0 where 0 is the addative identity..</p>
     *
     * @return The additive inverse for this field element.
     */
    public T addInv();


    /**
     * Computes the conjugate of this field element.
     * @return The conjugate of this field element.
     */
    public T conj();


    /**
     * <p>Computes the absolute value of this field element.</p>
     * <p>By default, this is implemented as {@code return }{@link #mag()}{@code ;}</p>
     *
     * @return The absolute value of this field element.
     */
    default double abs() {
        return mag();
    }


    /**
     * Computes the magnitude of this field element.
     * @return The magniitude of this field element.
     */
    public double mag();


    /**
     * Computes the square root of this field element.
     * @return The square root of this field element.
     */
    public T sqrt();


    /**
     * Compares this element of the ordered field with {@code b}.
     * @param b Second elemetn of the ordered field.
     * @return An int value:
     * <ul>
     *     <li>0 if this field element is equal to {@code b}.</li>
     *     <li>< 0 if this field element is less than {@code b}.</li>
     *     <li>> 0 if this field element is greater than {@code b}.</li>
     *     Hence, this method returns zero if and only if the two field elemetns are equal, a negative value if and only the field
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
