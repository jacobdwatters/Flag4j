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

package org.flag4j.algebraic_structures;


import java.util.Objects;

/**
 * <p>Instances of this class can be used to store a pair of values (i.e. a 2-tuple).</p>
 * <p>Pairs are immutable.</p>
 *
 * @param <T> The type of the elements of the pair.
 */
public class Pair<T> {
    /**
     * First value in the pair.
     */
    private final T first;
    /**
     * Second value in the pair.
     */
    private final T second;


    /**
     * Constructs a pair with the specified entries.
     * @param first First entry in the pair.
     * @param second Second entry in the pair.
     */
    public Pair(T first, T second) {
        this.first = first;
        this.second = second;
    }


    /**
     * Gets the first entry of this pair.
     * @return The first entry of this pair.
     */
    public T getFirst() {
        return first;
    }


    /**
     * Gets the second entry of this pair.
     * @return The second entry of this pair.
     */
    public T getSecond() {
        return second;
    }


    /**
     * Checks an object is equal to this pair.
     * @param o Object to check for equality with this pair.
     * @return True if {@code o} is type {@link Pair} and both the first and second items of this pair and {@code o} are equal as
     * defined by {@link Objects#equals(Object, Object)}.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Pair<?> pair = (Pair<?>) o;

        return Objects.equals(first, pair.first) &&
                Objects.equals(second, pair.second);
    }


    @Override
    public int hashCode() {
        return Objects.hash(first, second);
    }
}
