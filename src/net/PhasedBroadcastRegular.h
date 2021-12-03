
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_NET_PHASEDBROADCASTREGULAR_H
#define HEMELB_NET_PHASEDBROADCASTREGULAR_H

#include "net/PhasedBroadcast.h"

namespace hemelb
{
  namespace net
  {
    /**
     * PhasedBroadcastRegular - a class for performing phased broadcasts starting at regular
     * intervals. A longer description is given in PhasedBroadcast.h.
     */
    template<bool initialAction = false, unsigned splay = 1, unsigned overlap = 0, bool goDown = true, bool goUp = true>
    class PhasedBroadcastRegular : public PhasedBroadcast<initialAction, splay, overlap, goDown, goUp>
    {
      public:
        /**
         * Constructor that calls the base class's constructor.
         *
         * @param iNet
         * @param iSimState
         * @param spreadFactor
         * @return
         */
        PhasedBroadcastRegular(Net * iNet, const lb::SimulationState * iSimState, unsigned int spreadFactor) :
            base(iNet, iSimState, spreadFactor)
        {

        }

        /**
         * Function that requests all the communications from the Net object.
         */
        void RequestComms()
        {
          const unsigned long iCycleNumber = Get0IndexedIterationNumber();
          const unsigned long firstAscent = base::GetFirstAscending();
          const unsigned long firstDescent = base::GetFirstDescending();

          // Nothing to do for initial action case.

          // Next, deal with the case of a cycle with an initial pass down the tree.
          if (goDown)
          {
            if (iCycleNumber >= firstDescent && iCycleNumber < firstAscent)
            {
              unsigned long sendOverlap;
              unsigned long receiveOverlap;

              if (base::GetSendChildrenOverlap(iCycleNumber - firstDescent, &sendOverlap))
              {
                ProgressToChildren(sendOverlap);
              }

              if (base::GetReceiveParentOverlap(iCycleNumber - firstDescent, &receiveOverlap))
              {
                ProgressFromParent(receiveOverlap);
              }
            }
          }

          // And deal with the case of a cycle with a pass up the tree.
          if (goUp)
          {
            if (iCycleNumber >= firstAscent)
            {
              unsigned long sendOverlap;
              unsigned long receiveOverlap;

              if (base::GetSendParentOverlap(iCycleNumber - firstAscent, &sendOverlap))
              {
                ProgressToParent(sendOverlap);
              }

              if (base::GetReceiveChildrenOverlap(iCycleNumber - firstAscent, &receiveOverlap))
              {
                ProgressFromChildren(receiveOverlap);
              }
            }
          }
        }

        /**
         * Function called after send begin but before receives are known to be completed. The
         * action performed is limited to the InitialAction, if used.
         */
        void PreReceive()
        {
          // The only thing to do while waiting is the initial action.
          if (initialAction)
          {
            if (Get0IndexedIterationNumber() == 0)
            {
              InitialAction();
            }
          }
        }

        /**
         * Function to be called after the Receives have completed, where the
         * data is used.
         */
        void PostReceive()
        {
          const unsigned long iCycleNumber = Get0IndexedIterationNumber();
          const unsigned long firstAscent =
              PhasedBroadcast<initialAction, splay, overlap, goDown, goUp>::GetFirstAscending();
          const unsigned long traversalLength =
              PhasedBroadcast<initialAction, splay, overlap, goDown, goUp>::GetTraverseTime();


          // Propagate this info every 1000 timesteps - Check again in the future!!! To do!!!
          if(this->mSimState->GetTimeStep() % 1000 == 0){
            //printf("Rank: %d, PhasedBroadcastRegular -> PostReceive, Time: %lu, iCycleNumber: %lu, firstAscent: %lu, traversalLength: %lu  \n\n", this->mNet->GetCommunicator().Rank(), \
              this->mSimState->GetTimeStep(), iCycleNumber, firstAscent, traversalLength);

            // Deal with the case of a cycle with an initial pass down the tree.
            if (goDown)
            {
              // Enters here indeed!!!  printf("Within the goDown loop, Time: %lu\n\n", this->mSimState->GetTimeStep());
              const unsigned long firstDescent =
                  PhasedBroadcast<initialAction, splay, overlap, goDown, goUp>::GetFirstDescending();

              if (iCycleNumber >= firstDescent && iCycleNumber < firstAscent)
              {
                unsigned long receiveOverlap;

                if (base::GetReceiveParentOverlap(iCycleNumber - firstDescent, &receiveOverlap))
                {
                  PostReceiveFromParent(receiveOverlap);
                }

                // If we're halfway through the programme, all top-down changes have occurred and
                // can be applied on all nodes at once safely.
                if ( (iCycleNumber - firstDescent) == (traversalLength - 1))
                {
                  //printf("Calling Effect - Time: %lu \n\n", this->mSimState->GetTimeStep());
                  Effect();
                }
              }
            }

            // Deal with the case of a cycle with a pass back up the tree.
            if (goUp)
            {
              // Enters here indeed!!! printf("Within the goUp loop, Time: %lu\n\n", this->mSimState->GetTimeStep());

              if (iCycleNumber >= firstAscent)
              {
                unsigned long receiveOverlap, sendOverlap;

                if (base::GetReceiveChildrenOverlap(iCycleNumber - firstAscent, &receiveOverlap))
                {
                  PostReceiveFromChildren(receiveOverlap);
                }

                if (base::GetSendParentOverlap(iCycleNumber - firstAscent, &sendOverlap))
                {
                  //printf("Rank: %d, Calling PostSendToParent - Time: %lu \n\n", this->mNet->GetCommunicator().Rank(), this->mSimState->GetTimeStep());
                  PostSendToParent(sendOverlap);

                  // IZ: Added Feb 2021
                  //   to force setting the stability status in mSimState->SetStability((Stability) mDownwardsStability);
                  Effect();
                  //
                }

              }
            }


            // If this node is the root of the tree and we've just finished the upwards half, it
            // must act.
            if (iCycleNumber == (base::GetRoundTripLength() - 1)
                && this->mNet->GetCommunicator().Rank() == 0)
            {
              //  printf("Rank: %d, Calling TopNodeAction - Time: %lu \n\n", this->mNet->GetCommunicator().Rank(), this->mSimState->GetTimeStep());
              TopNodeAction(); // Called only for rank=0 : sets mDownwardsStability to the value mUpwardsStability
              // IZ: Added Feb 2021
              // Effect();
            }

            /*
            // Get the values of the stability reported here (actually it didn't pass the stability value to this->mSimState->GetStability() yet):
            int stability_test = this->mSimState->GetStability();
            printf("Inside the loop - Time: %lu - Stability reported: %d \n\n", this->mSimState->GetTimeStep(), this->mSimState->GetStability());
            */
          } // Closes the if (iSimState->GetTimeStep()%1000 == 0) statement
        }

        /**
         * Returns the number of the iteration, as an integer between inclusive-0 and
         * exclusive-2 * (the tree depth)
         */
        unsigned long Get0IndexedIterationNumber() const
        {
          if (base::GetTreeDepth() > 0)
          {
            unsigned long stepsPassed = base::mSimState->Get0IndexedTimeStep();

            return stepsPassed % base::GetRoundTripLength();
          }
          else
          {
            return 0;
          }
        }

      protected:

        // Typedef for the base class type, for convenience.
        typedef PhasedBroadcast<initialAction, splay, overlap, goDown, goUp> base;

        /**
         * Overridable function for the initial action performed by a node at the beginning of the
         * cycle. Only has an effect if the template paramter initialAction is true.
         */
        virtual void InitialAction()
        {

        }

        /**
         * Overridable function for when a node has to receive from its children in the tree.
         *
         * Use ReceiveFromChildren to do this. The parameter splayNumber is 0 indexed and less
         * than splay.
         */
        virtual void ProgressFromChildren(unsigned long splayNumber)
        {

        }

        /**
         * Overridable function for when a node has to receive from its parent in the tree.
         *
         * Use ReceiveFromParent to do this. The parameter splayNumber is 0 indexed and less
         * than splay.
         */
        virtual void ProgressFromParent(unsigned long splayNumber)
        {

        }

        /**
         * Overridable function for when a node has to send to its children in the tree.
         *
         * Use SendToChildren to do this. The parameter splayNumber is 0 indexed and less
         * than splay.
         */
        virtual void ProgressToChildren(unsigned long splayNumber)
        {

        }

        /**
         * Overridable function for when a node has to send to its parent in the tree.
         *
         * Use SendToParent to do this. The parameter splayNumber is 0 indexed and less
         * than splay.
         */
        virtual void ProgressToParent(unsigned long splayNumber)
        {

        }

        /**
         * Overridable function, called by a node after data has been received from its children.
         * The parameter splayNumber is 0 indexed and less than splay.
         */
        virtual void PostReceiveFromChildren(unsigned long splayNumber)
        {

        }

        /**
         * Overridable function, called by a node after data has been sent to its parent.
         *
         * @param splayNumber The parameter splayNumber is 0 indexed and less than splay.
         */
        virtual void PostSendToParent(unsigned long splayNumber)
        {
        }

        /**
         * Overridable function, called by a node after data has been received from its parent. The
         * parameter splayNumber is 0 indexed and less than splay.
         */
        virtual void PostReceiveFromParent(unsigned long splayNumber)
        {

        }

        /**
         * Action taken when upwards-travelling data reaches the top node.
         */
        virtual void TopNodeAction()
        {

        }

        /**
         * Action taken by all nodes when downwards-travelling data has been sent to every node.
         */
        virtual void Effect()
        {

        }

#ifdef HEMELB_USE_GPU
        virtual void call_Stability_tester_GPU()
        {

        }
#endif

    };
  }
}

#endif /* HEMELB_NET_PHASEDBROADCASTREGULAR_H */
