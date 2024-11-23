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


import org.flag4j.algebraic_structures.fields.RealFloat64;

/**
 * <p>A real number backed by a 16-bit integer number. Immutable</p>
 *
 * <p>This class wraps the primitive short type.</p>
 */
public class RealInt16 implements Ring<RealInt16> {

    /**
     * The numerical value -1.
     */
    public final static RealInt16 NEGATIVE_ONE = new RealInt16(-1);
    /**
     * The numerical value 0.
     */
    public final static RealInt16 ZERO = new RealInt16(0);
    /**
     * The numerical value 1.
     */
    public final static RealInt16 ONE = new RealInt16(1);
    /**
     * The numerical value 2.
     */
    public final static RealInt16 TWO = new RealInt16(2);
    /**
     * The numerical value 3.
     */
    public final static RealInt16 THREE = new RealInt16(3);
    /**
     * The numerical value 10.
     */
    public final static RealInt16 TEN = new RealInt16(10);


    /**
     * Numerical value of field element.
     */
    private final short value;


    /**
     * Constructs a real 16-bit floating point number.
     * @param value Value of the 16-bit integer number.
     */
    public RealInt16(short value) {
        this.value = value;
    }


    /**
     * Constructs a real 16-bit floating point number by casting a 32-bit int.
     * @param value Value of the 16-bit integer number.
     */
    public RealInt16(int value) {
        this.value = (short) value;
    }


    /**
     * Constructs a real 16-bit floating point number.
     * @param value Value of the 16-bit integer number.
     */
    public RealInt16(double value) {
        this.value = (short) value;
    }


    /**
     * Constructs a real 16-bit integer number by casting a real 64-bit floating point.
     * @param value Value of the 16-bit integer number.
     */
    public RealInt16(RealFloat64 value) {
        this.value = (short) value.getValue();
    }


    /**
     * Gets the value of this field element.
     * @return The value of this field element.
     */
    public float getValue() {
        return value;
    }


    /**
     * Sums two elements of this field (associative and commutative).
     *
     * @param b Second field element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public RealInt16 add(RealInt16 b) {
        return new RealInt16(this.value + b.value);
    }


    /**
     * Computes difference of two elements of this field.
     *
     * @param b Second field element in difference.
     *
     * @return The difference of this field element and {@code b}.
     */
    @Override
    public RealInt16 sub(RealInt16 b) {
        return new RealInt16(this.value - b.value);
    }


    /**
     * Multiplies two elements of this field (associative and commutative).
     *
     * @param b Second field element in product.
     *
     * @return The product of this field element and {@code b}.
     */
    @Override
    public RealInt16 mult(RealInt16 b) {
        return new RealInt16(this.value * b.value);
    }


    /**
     * <p>Checks if this value is an additive identity for this semi-ring.</p>
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semi-ring.</p>
     *
     * @return True if this value is an additive identity for this semi-ring. Otherwise, false.
     */
    @Override
    public boolean isZero() {
        return equals(ZERO);
    }


    /**
     * <p>Checks if this value is a multiplicative identity for this semi-ring.</p>
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semi-ring.</p>
     *
     * @return True if this value is a multiplicative identity for this semi-ring. Otherwise, false.
     */
    @Override
    public boolean isOne() {
        return equals(ONE);
    }


    /**
     * <p>Gets the additive identity for this semi-ring.</p>
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semi-ring.</p>
     *
     * @return The additive identity for this semi-ring.
     */
    @Override
    public RealInt16 getZero() {
        return ZERO;
    }


    /**
     * <p>Gets the multiplicative identity for this semi-ring.</p>
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semi-ring.</p>
     *
     * @return The multiplicative identity for this semi-ring.
     */
    @Override
    public RealInt16 getOne() {
        return ONE;
    }


    /**
     * <p>Computes the additive inverse for an element of this field.</p>
     *
     * <p>An element -x is an additive inverse for a filed element x if -x + x = 0 where 0 is the additive identity..</p>
     *
     * @return The additive inverse for this field element.
     */
    @Override
    public RealInt16 addInv() {
        return new RealInt16(-this.value);
    }


    /**
     * Computes the magnitude of this field element.
     *
     * @return The magnitude of this field element.
     */
    @Override
    public double mag() {
        return Math.abs(this.value);
    }


    /**
     * Evaluates the signum or sign function on a field element.
     *
     * @param a Value to evaluate signum function on.
     * @return The output of the signum function evaluated on {@code a}.
     */
    public static RealInt16 sgn(RealInt16 a) {
        return new RealInt16(Math.signum(a.value));
    }


    /**
     * Compares this element of the ordered field with {@code b}.
     *
     * @param b Second element of the ordered field.
     *
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
    @Override
    public int compareTo(RealInt16 b) {
        return Integer.compare(this.value, b.value);
    }


    /**
     * Converts this semiring value to an equivalent double value.
     *
     * @return A double value equivalent to this semiring element.
     */
    @Override
    public double doubleValue() {
        return value;
    }


    /**
     * Computes the product of all entire of specified array.
     * @param values Values to compute product of.
     * @return The product of all values in {@code values}.
     */
    public static RealInt16 prod(RealInt16... values) {
        if(values == null || values.length == 0) {
            throw new IllegalArgumentException("Values cannot be null or empty");
        }

        short prod = values[0].value;

        for(int i = 1, length=values.length; i < length; i++) {
            prod *= values[i].value;
        }

        return new RealInt16(prod);
    }


    /**
     * Computes the sum of all entire of specified array.
     * @param values Values to compute product of.
     * @return The sum of all values in {@code values}.
     */
    public static RealInt16 sum(RealInt16... values) {
        if(values == null || values.length == 0) {
            throw new IllegalArgumentException("Values cannot be null or empty");
        }

        short sum = values[0].value;

        for(int i = 1, length=values.length; i < length; i++) {
            sum += values[i].value;
        }

        return new RealInt16(sum);
    }


    /**
     * Checks if an object is equal to this Field element.
     * @param b Object to compare to this Field element.
     * @return True if the objects are the same or are both {@link RealInt16}'s and have equal values.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((RealInt16) b).value;
    }


    /**
     * Converts this field element to a string representation.
     * @return A string representation of this field element.
     */
    public String toString() {
        return String.valueOf(this.value);
    }


    @Override
    public int hashCode() {
        return Short.hashCode(value);
    }
}
