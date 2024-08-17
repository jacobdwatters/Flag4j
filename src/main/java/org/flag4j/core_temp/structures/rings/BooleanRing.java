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
 * <p>A semi-ring element backed by a boolean. Immutable</p>
 *
 * <p>This class wraps the primative boolean type.</p>
 */
public class BooleanRing implements SemiRing<BooleanRing> {

    /**
     * Boolean value of field element.
     */
    private final boolean value;


    /**
     * Constructs a boolean ring element.
     * @param value Value of the boolean ring element.
     */
    public BooleanRing(boolean value) {
        this.value = value;
    }


    /**
     * <p>Sums two elements of this semi-ring (associative and commutative).</p>
     *
     * <p>This is equivalent to logical OR</p>
     *
     * @param b Second semi-ring element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public BooleanRing add(BooleanRing b) {
        return new BooleanRing(value || b.value);
    }


    /**
     * Computes "exclusive OR" (i.e. xor) of elements of this semi-ring.
     *
     * @param b Second semi-ring element in xor.
     *
     * @return The xor of this element and {@code b}.
     */
    public BooleanRing xor(BooleanRing b) {
        return new BooleanRing(value ^ b.value);
    }


    /**
     * Multiplies two elements of this semi-ring (associative).
     * <p>This is equivalent to logical AND</p>
     *
     * @param b Second semi-ring element in product.
     *
     * @return The product of this semi-ring element and {@code b}.
     */
    @Override
    public BooleanRing mult(BooleanRing b) {
        return new BooleanRing(value && b.value);
    }


    /**
     * Compares this element of the semi-ring with {@code b}.
     *
     * @param b Second element of the semi-ring.
     *
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
    @Override
    public int compareTo(BooleanRing b) {
        return Boolean.compare(value, b.value);
    }
}
