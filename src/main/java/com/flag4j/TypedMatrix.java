package com.flag4j;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;

/**
 * Stores the type and shape of matrix object.
 * @param <T> The type of the entry of this matrix.
 */
public abstract class TypedMatrix<T> {

    /**
     * The type of this matrix.
     */
    public final MatrixTypes type;
    /**
     * The values of this matrix.
     */
    public T entries; // TODO: make protected
    /**
     * The number of rows in this matrix.
     */
    protected int m;
    /**
     * The number of columns in this matrix.
     */
    protected int n;


    /**
     * Constructs a typed matrix with a specified size.
     * @param type The type of this matrix.
     * @param m The number of rows in this matrix.
     * @param n The number of columns in this matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    protected TypedMatrix(MatrixTypes type, int m, int n) {
        if(m<0 || n<0) {
            throw new IllegalArgumentException(
                    ErrorMessages.negativeDimErrMsg(this.getShape())
            );
        }

        this.m = m;
        this.n = n;
        this.type = type;
    }


    protected TypedMatrix(MatrixTypes type, Shape shape) {
        if(shape.getRank() != 2){
            throw new IllegalArgumentException(ErrorMessages.shapeRankErr(2, shape.getRank()));
        }

        this.m = shape.get(0);
        this.n = shape.get(1);
        this.type = type;
    }


    /**
     * Gets the number of rows in a matrix.
     *
     * @return The number of rows in this matrix.
     */
    public int numRows() {
        return m;
    }


    /**
     * Gets the number of columns in a matrix.
     *
     * @return The number of columns in this matrix.
     */
    public int numCols() {
        return n;
    }



    /**
     * Gets the shape of this tensor.
     * @return A shape object describing the
     */
    public Shape getShape() {
        return new Shape(m, n);
    }


    /**
     * Checks if this tensor is empty. That is, if this tensor contains no elements. This is equivalent to <code>numRows==0 and
     * numCols==0</code>.
     * @return True if this tensor is empty (i.e. contains no elements). Otherwise, returns false.
     */
    public boolean isEmpty() {
        return m==0 && n==0;
    }


    /**
     * Checks if two matrices have the same shape.
     * @param A First matrix to compare.
     * @param B Second matrix to compare.
     * @return True if matrix A and matrix B have the same shape. Otherwise, returns false.
     */
    public static boolean sameShape(TypedMatrix A, TypedMatrix B) {
        return A.getShape().equals(B.getShape());
    }


    /**
     * Checks if two matrices have the same length along a specified axis.
     * @param A First matrix to compare.
     * @param B Second matrix to compare.
     * @param axis The axis along which to compare the lengths of the two matrices.<br>
     *             - If axis=0, then the number of rows is compared.<br>
     *             - If axis=1, then the number of columns is compared.
     * @return True if matrix A and Matrix B have the same length along the specified axis. Otherwise, returns false.
     * @throws IllegalArgumentException If axis is not zero or one.
     */
    public static boolean sameLength(TypedMatrix A, TypedMatrix B, int axis) {
        if(axis!=Axis2D.row() || axis!=Axis2D.col()) {
            // Then this is an unspecified axis
            throw new IllegalArgumentException(
                    ErrorMessages.axisErr(axis, Axis2D.allAxes())
            );
        }

        boolean result = false;

        if(axis == Axis2D.row()) {
            result = A.m == B.m;

        } else if(axis == Axis2D.col()) {
            result = A.n == B.n;
        }

        return result;
    }
}
