// this is a virtual class


class RandSketchingManager {
  public: 
    // constructor
 
    RandSketchingManager();
 
    // destructor
    virtual ~RandSketchingManager();

    // Actual sketching process
    virtual int Theta(vector_type* input, vector_type* output);
   
    // Setup the parameters, sampling matrices, permuations, etc
    virtual int setup(index_type n, index_type k);
    virtual int reset();
  
  private:
    index_type n_;// size of base vector
    index_type k_; // size of sketched vector
};
