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

package org.flag4j.algebraic_structures.semirings;


/**
 * <p>A semiring element backed by a boolean. Immutable</p>
 *
 * <p>This class wraps the primitive boolean type.</p>
 */
public class Bool implements Semiring<Bool> {
    // Constants provided for convenience.

    /**
     * The boolean value true.
     */
    final public static Bool ONE = new Bool(true);
    /**
     * The boolean value true.
     */
    final public static Bool TRUE = ONE;
    /**
     * The boolean value false.
     */
    final public static Bool ZERO = new Bool(false);
    /**
     * The boolean value false.
     */
    final public static Bool FALSE = ZERO;


    /**
     * Boolean value of field element.
     */
    private final boolean value;


    /**
     * Constructs a boolean ring element.
     * @param value Value of the boolean ring element.
     */
    public Bool(boolean value) {
        this.value = value;
    }


    /**
     * Constructs a boolean ring element from an integer.
     * @param value Value of the boolean ring element.
     */
    public Bool(int value) {
        if(value == 0) this.value = false;
        else if(value == 1) this.value = true;
        else throw new IllegalArgumentException("Cannot convert int value " + value + " to boolean. Must be 1 or 0.");
    }


    /**
     * <p>Sums two elements of this semiring (associative and commutative).
     *
     * <p>This is equivalent to logical OR.
     *
     * @param b Second semiring element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public Bool add(Bool b) {
        return new Bool(value || b.value);
    }


    /**
     * <p>Computes the logical OR between two boolean semiring elements.
     * <p>Same as {@link #add(Bool)}
     * @param b Second element in the logical OR.
     * @return Logical OR of this boolean semiring element and {@code b}.
     */
    public Bool or(Bool b) {
        return add(b);
    }


    /**
     * Computes "exclusive OR" (i.e. XOR) of elements of this semiring.
     *
     * @param b Second semiring element in XOR.
     *
     * @return The XOR of this element and {@code b}.
     */
    public Bool xor(Bool b) {
        return new Bool(value ^ b.value);
    }


    /**
     * <p>Multiplies two elements of this semiring (associative).
     * <p>This is equivalent to logical AND.
     *
     * @param b Second semiring element in product.
     *
     * @return The product of this semiring element and {@code b}.
     */
    @Override
    public Bool mult(Bool b) {
        return new Bool(value && b.value);
    }


    /**
     * <p>Computes the logical and between two boolean semiring elements.
     * <p>Same as {@link #mult(Bool)}
     * @param b Second element in the logical AND.
     * @return Logical AND of this boolean semiring element and {@code b}.
     */
    public Bool and(Bool b) {
        return mult(b);
    }


    /**
     * Computes the logical NOT (i.e. negation) of this boolean semiring element.
     * @return The logical NOT of this boolean semiring element.
     */
    public Bool not() {
        return new Bool(!value);
    }


    /**
     * <p>Checks if this value is an additive identity for this semiring.</p>
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.</p>
     *
     * @return True if this value is an additive identity for this semiring. Otherwise, false.
     */
    @Override
    public boolean isZero() {
        return equals(ZERO);
    }


    /**
     * <p>Checks if this value is a multiplicative identity for this semiring.</p>
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.</p>
     *
     * @return True if this value is a multiplicative identity for this semiring. Otherwise, false.
     */
    @Override
    public boolean isOne() {
        return equals(ONE);
    }


    /**
     * <p>Gets the additive identity for this semiring.</p>
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.</p>
     *
     * @return The additive identity for this semiring.
     */
    @Override
    public Bool getZero() {
        return ZERO;
    }


    /**
     * <p>Gets the multiplicative identity for this semiring.</p>
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.</p>
     *
     * @return The multiplicative identity for this semiring.
     */
    @Override
    public Bool getOne() {
        return ONE;
    }


    /**
     * Compares this element of the semiring with {@code b}.
     *
     * @param b Second element of the semiring.
     *
     * @return An int value:
     * <ul>
     *     <li>0 if this semiring element is equal to {@code b}.</li>
     *     <li>< 0 if this semiring element is less than {@code b}.</li>
     *     <li>> 0 if this semiring element is greater than {@code b}.</li>
     *     Hence, this method returns zero if and only if the two semiring elements are equal, a negative value if and only the semiring
     *     element it was called on is less than {@code b} and positive if and only if the semiring element it was called on is greater
     *     than {@code b}.
     * </ul>
     */
    @Override
    public int compareTo(Bool b) {
        return Boolean.compare(value, b.value);
    }


    /**
     * Converts this semiring value to an equivalent double value.
     *
     * @return A double value equivalent to this semiring element.
     */
    @Override
    public double doubleValue() {
        return (value) ? 1.0 : 0.0;
    }


    /**
     * Checks if an object is equal to this semiring element.
     * @param b Object to compare to this semiring element.
     * @return True if the objects are the same or are both {@link Bool}'s and have equal values.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((Bool) b).value;
    }


    @Override
    public int hashCode() {
        return Boolean.hashCode(value);
    }
}
