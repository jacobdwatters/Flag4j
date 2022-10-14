package com.flag4j.linalg;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;

/**
 * This class provides several methods, which are not provided as constructors, that are useful for creating matrices.
 */
public class CreateMatrix {


    /**
     * Constructs a diagonal matrix whose non-zero components, i.e. the components of the principle diagonal are specified.
     * The length of the specified values will determine the size of the resultant matrix.
     * @param values Values to use as diagonal components of the resultant matrix.
     * @return A square, diagonal matrix whose non-zero components are equivalent to those specified by 'values'.
     */
    SparseMatrix toDiag(int[] values){/*TODO:*/return null;}


    /**
     * Constructs a diagonal matrix whose non-zero components, i.e. the components of the principle diagonal are specified.
     * The length of the specified values will determine the size of the resultant matrix.
     * @param values Values to use as diagonal components of the resultant matrix.
     * @return A square, diagonal matrix whose non-zero components are equivalent to those specified by 'values'.
     */
    SparseMatrix toDiag(double[] values){/*TODO:*/return null;}


    /**
     * Constructs a diagonal matrix whose non-zero components, i.e. the components of the principle diagonal are specified.
     * The length of the specified values will determine the size of the resultant matrix.
     * @param values Values to use as diagonal components of the resultant matrix.
     * @return A square, diagonal matrix whose non-zero components are equivalent to those specified by 'values'.
     */
    SparseMatrix toDiag(Vector values){/*TODO:*/return null;}


    /**
     * Constructs a diagonal matrix whose non-zero components, i.e. the components of the principle diagonal are specified.
     * The length of the specified values will determine the size of the resultant matrix.
     * @param values Values to use as diagonal components of the resultant matrix.
     * @return A square, diagonal matrix whose non-zero components are equivalent to those specified by 'values'.
     */
    SparseMatrix toDiag(SparseVector values){/*TODO:*/return null;}


    /**
     * Constructs a diagonal matrix whose non-zero components, i.e. the components of the principle diagonal are specified.
     * The length of the specified values will determine the size of the resultant matrix.
     * @param values Values to use as diagonal components of the resultant matrix.
     * @return A square, diagonal matrix whose non-zero components are equivalent to those specified by 'values'.
     */
    SparseCMatrix toDiag(CNumber[] values){/*TODO:*/return null;}


    /**
     * Constructs a diagonal matrix whose non-zero components, i.e. the components of the principle diagonal are specified.
     * The length of the specified values will determine the size of the resultant matrix.
     * @param values Values to use as diagonal components of the resultant matrix.
     * @return A square, diagonal matrix whose non-zero components are equivalent to those specified by 'values'.
     */
    SparseCMatrix toDiag(CVector values){/*TODO:*/return null;}


    /**
     * Constructs a diagonal matrix whose non-zero components, i.e. the components of the principle diagonal are specified.
     * The length of the specified values will determine the size of the resultant matrix.
     * @param values Values to use as diagonal components of the resultant matrix.
     * @return A square, diagonal matrix whose non-zero components are equivalent to those specified by 'values'.
     */
    SparseCMatrix toDiag(SparseCMatrix values){/*TODO:*/return null;}
}
