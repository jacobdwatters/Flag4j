package com.flag4j;


/**
 * This interface specifies comparisons which all vectors should implement.
 *
 * @param <T> Vector type.
 * @param <U> Dense vector type.
 * @param <V> Sparse vector type.
 * @param <W> Complex vector type.
 * @param <Y> Real vector type.
 * @param <X> Vector entry type.
 */
public interface VectorComparisonsMixin<T, U, V, W, Y, X extends Number> extends TensorComparisonsMixin<T, U, V, W, Y, X> {

    /**
     * Checks if two vectors are equal (element-wise). This method <b>DOES</b> take into consideration the
     * orientation of the vectors.
     * @param b Second vector in the equality.
     * @return True if this vector and vector b are equivalent element-wise. Otherwise, returns false.
     */
    boolean equals(Vector b);


    /**
     * Checks if two vectors are equal (element-wise). This method <b>DOES</b> take into consideration the
     * orientation of the vectors.
     * @param b Second vector in the equality.
     * @return True if this vector and vector b are equivalent element-wise. Otherwise, returns false.
     */
    boolean equals(SparseVector b);


    /**
     * Checks if two vectors are equal (element-wise). This method <b>DOES</b> take into consideration the
     * orientation of the vectors.
     * @param b Second vector in the equality.
     * @return True if this vector and vector b are equivalent element-wise. Otherwise, returns false.
     */
    boolean equals(CVector b);


    /**
     * Checks if two vectors are equal (element-wise). This method <b>DOES</b> take into consideration the
     * orientation of the vectors.
     * @param b Second vector in the equality.
     * @return True if this vector and vector b are equivalent element-wise. Otherwise, returns false.
     */
    boolean equals(SparseCVector b);


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    boolean sameShape(Vector b);


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    boolean sameShape(SparseVector b);


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    boolean sameShape(CVector b);


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    boolean sameShape(SparseCVector b);




    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    boolean sameLength(Vector b);


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    boolean sameLength(SparseVector b);


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    boolean sameLength(CVector b);


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    boolean sameLength(SparseCVector b);
}
