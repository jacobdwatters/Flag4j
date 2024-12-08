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

/**
 * <p>Provides algebraic structures such as semirings, rings, and fields,
 * along with their concrete implementations.
 *
 * <p>This module defines interfaces representing algebraic structures:
 * <ul>
 *   <li>{@link Semiring}</li>
 *   <li>{@link Ring}</li>
 *   <li>{@link Field}</li>
 * </ul>
 * and provides concrete implementations for various data types:
 * <ul>
 *   <li>{@link org.flag4j.algebraic_structures.Bool}</li>
 *   <li>{@link org.flag4j.algebraic_structures.RealInt16}</li>
 *   <li>{@link org.flag4j.algebraic_structures.RealInt32}</li>
 *   <li>{@link org.flag4j.algebraic_structures.Real32}</li>
 *   <li>{@link org.flag4j.algebraic_structures.Real64}</li>
 *   <li>{@link org.flag4j.algebraic_structures.Complex64}</li>
 *   <li>{@link org.flag4j.algebraic_structures.Complex128}</li>
 * </ul>
 *
 * <p>These abstractions allow for generic programming with mathematical structures,
 * enabling algorithms to work over any data type that adheres to the specified algebraic structure.
 */
package org.flag4j.algebraic_structures;